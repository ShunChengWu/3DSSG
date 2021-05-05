if __name__ == '__main__' and __package__ is None:
    from os import sys, path
    sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

import os,sys,time,math,torch
import numpy as np
from torch_geometric.nn.conv import MessagePassing

def read_txt_to_list(file):
    output = [] 
    with open(file, 'r') as f: 
        for line in f: 
            entry = line.rstrip().lower() 
            output.append(entry) 
    return output

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                      [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                      [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

def gen_descriptor(pts:torch.tensor):
    '''
    centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths
    [3, 3, 3, 1, 1]
    '''
    assert pts.ndim==2
    assert pts.shape[-1]==3
    # centroid [n, 3]
    centroid_pts = pts.mean(0) 
    # # std [n, 3]
    std_pts = pts.std(0)
    # dimensions [n, 3]
    segment_dims = pts.max(dim=0)[0] - pts.min(dim=0)[0]
    # volume [n, 1]
    segment_volume = (segment_dims[0]*segment_dims[1]*segment_dims[2]).unsqueeze(0)
    # length [n, 1]
    segment_lengths = segment_dims.max().unsqueeze(0)
    return torch.cat([centroid_pts,std_pts,segment_dims,segment_volume,segment_lengths],dim=0)


class Gen_edge_descriptor(MessagePassing):#TODO: move to model
    """ A sequence of scene graph convolution layers  """
    def __init__(self, flow="source_to_target"):
        super().__init__(flow=flow)
    def forward(self, descriptor, edges_indices):
        size = self.__check_input__(edges_indices, None)
        coll_dict = self.__collect__(self.__user_args__,edges_indices,size, {"x":descriptor})
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        edge_feature = self.message(**msg_kwargs)
        return edge_feature
    
    def message(self, x_i, x_j):
        # source_to_target
        # (j, i)
        # 0-2: centroid, 3-5: std, 6-8:dims, 9:volume, 10:length
        # to
        # 0-2: offset centroid, 3-5: offset std, 6-8: dim log ratio, 9: volume log ratio, 10: length log ratio
        edge_feature = torch.zeros_like(x_i)
        # centroid offset
        edge_feature[:,0:3] = x_i[:,0:3]-x_j[:,0:3]
        # std  offset
        edge_feature[:,3:6] = x_i[:,3:6]-x_j[:,3:6]
        # dim log ratio
        edge_feature[:,6:9] = torch.log(x_i[:,6:9] / x_j[:,6:9])
        # volume log ratio
        edge_feature[:,9] = torch.log( x_i[:,9] / x_j[:,9])
        # length log ratio
        edge_feature[:,10] = torch.log( x_i[:,10] / x_j[:,10])
        # edge_feature, *_ = self.ef(edge_feature.unsqueeze(-1))
        return edge_feature.unsqueeze(-1)

       
def pytorch_count_params(model, trainable=True):
    "count number trainable parameters in a pytorch model"
    s = 0
    for p in model.parameters():
        if trainable:
            if not p.requires_grad: continue
        try:
            s += p.numel()
        except:
            pass
    return s


class Progbar(object):
    """Displays a progress bar.

    Arguments:
        target: Total number of steps expected, None if unknown.
        width: Progress bar width on screen.
        verbose: Verbosity mode, 0 (silent), 1 (verbose), 2 (semi-verbose)
        stateful_metrics: Iterable of string names of metrics that
            should *not* be averaged over time. Metrics in this list
            will be displayed as-is. All others will be averaged
            by the progbar before display.
        interval: Minimum visual progress update interval (in seconds).
    """

    def __init__(self, target, width=25, verbose=1, interval=0.05,
                  stateful_metrics=None):
        self.target = target
        self.width = width
        self.verbose = verbose
        self.interval = interval
        if stateful_metrics:
            self.stateful_metrics = set(stateful_metrics)
        else:
            self.stateful_metrics = set()

        self._dynamic_display = ((hasattr(sys.stdout, 'isatty') and
                                  sys.stdout.isatty()) or
                                  'ipykernel' in sys.modules or
                                  'posix' in sys.modules)
        self._total_width = 0
        self._seen_so_far = 0
        # We use a dict + list to avoid garbage collection
        # issues found in OrderedDict
        self._values = {}
        self._values_order = []
        self._start = time.time()
        self._last_update = 0

    def update(self, current, values=None, silent=False):
        """Updates the progress bar.

        Arguments:
            current: Index of current step.
            values: List of tuples:
                `(name, value_for_last_step)`.
                If `name` is in `stateful_metrics`,
                `value_for_last_step` will be displayed as-is.
                Else, an average of the metric over time will be displayed.
        """
        values = values or []
        for k, v in values:
            if k not in self._values_order:
                self._values_order.append(k)
            if k not in self.stateful_metrics:
                if k not in self._values:
                    self._values[k] = [v * (current - self._seen_so_far),
                                        current - self._seen_so_far]
                else:
                    self._values[k][0] += v * (current - self._seen_so_far)
                    self._values[k][1] += (current - self._seen_so_far)
            else:
                self._values[k] = v
        self._seen_so_far = current

        now = time.time()
        info = ' - %.0fs' % (now - self._start)
        if self.verbose == 1:
            if (now - self._last_update < self.interval and
                    self.target is not None and current < self.target):
                return

            prev_total_width = self._total_width
            if not silent:
                if self._dynamic_display:
                    sys.stdout.write('\b' * prev_total_width)
                    sys.stdout.write('\r')
                else:
                    sys.stdout.write('\n')

            if self.target is not None:
                numdigits = int(np.floor(np.log10(self.target))) + 1
                barstr = '%%%dd/%d [' % (numdigits, self.target)
                bar = barstr % current
                prog = float(current) / self.target
                prog_width = int(self.width * prog)
                if prog_width > 0:
                    bar += ('=' * (prog_width - 1))
                    if current < self.target:
                        bar += '>'
                    else:
                        bar += '='
                bar += ('.' * (self.width - prog_width))
                bar += ']'
            else:
                bar = '%7d/Unknown' % current

            self._total_width = len(bar)
            if not silent:
                sys.stdout.write(bar)

            if current:
                time_per_unit = (now - self._start) / current
            else:
                time_per_unit = 0
            if self.target is not None and current < self.target:
                eta = time_per_unit * (self.target - current)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (eta // 3600,
                                                    (eta % 3600) // 60,
                                                    eta % 60)
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info = ' - ETA: %s' % eta_format
            else:
                if time_per_unit >= 1:
                    info += ' %.0fs/step' % time_per_unit
                elif time_per_unit >= 1e-3:
                    info += ' %.0fms/step' % (time_per_unit * 1e3)
                else:
                    info += ' %.0fus/step' % (time_per_unit * 1e6)

            for k in self._values_order:
                info += ' - %s:' % k
                if isinstance(self._values[k], list):
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if abs(avg) > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                else:
                    info += ' %s' % self._values[k]

            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += (' ' * (prev_total_width - self._total_width))

            if self.target is not None and current >= self.target:
                info += '\n'

            if not silent:
                sys.stdout.write(info)
                sys.stdout.flush()

        elif self.verbose == 2:
            if self.target is None or current >= self.target:
                for k in self._values_order:
                    info += ' - %s:' % k
                    avg = np.mean(self._values[k][0] / max(1, self._values[k][1]))
                    if avg > 1e-3:
                        info += ' %.4f' % avg
                    else:
                        info += ' %.4e' % avg
                info += '\n'
                if not silent:
                    sys.stdout.write(info)
                    sys.stdout.flush()

        self._last_update = now

    def add(self, n, values=None,silent=False):
        self.update(self._seen_so_far + n, values,silent=silent)

def check(x,y):
    x = x if isinstance(x, list) or isinstance(x, tuple) else [x]
    y = y if isinstance(y, list) or isinstance(y, tuple) else [y]
    [np.testing.assert_allclose(x[i].flatten(), y[i].flatten(), rtol=1e-03, atol=1e-05) for i in range(len(x))]
        
def export(model:torch.nn.Module, inputs:list,pth:str, input_names:list, output_names:list, dynamic_axes:dict):
    import onnxruntime as ort
    inputs = inputs if isinstance(inputs, list) or isinstance(inputs, tuple) else [inputs]
    torch.onnx.export(model = model, args = tuple(inputs), f=pth,
              verbose=False,export_params=True,
              do_constant_folding=True,
              input_names=input_names, output_names=output_names,
              dynamic_axes=dynamic_axes,opset_version=12)
    with torch.no_grad():
        model.eval()
        sess = ort.InferenceSession(pth)
        x = model(*inputs)
        ins = {input_names[i]: inputs[i].numpy() for i in range(len(inputs))}
        y = sess.run(None, ins)
        check(x,y)
        
        inputs = [torch.cat([input,input],dim=0) for input in inputs]
        x = model(*inputs)
        ins = {input_names[i]: inputs[i].numpy() for i in range(len(inputs))}
        y = sess.run(None, ins)
        check(x,y)

def get_tensorboard_logs(pth_log):
    for (dirpath, dirnames, filenames) in os.walk(pth_log):
        break
    l = list()
    for filename in filenames:
        if filename.find('events') >= 0: l.append(filename)
    return l
        
def create_dir(dir):
    from pathlib import Path
    Path(dir).mkdir(parents=True, exist_ok=True)