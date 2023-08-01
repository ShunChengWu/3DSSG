
# Structure
kId: keyfame index
imgBoxData: [x1/img_w,y1/img_h,x2/img_w,y2/img_h, occlu_level] 


```
proposals.h5
{
    'args': arguments.
    'label_type': label type
    scan_id: {
        'nodes': {
            oid: # object Index
                data = [keyframe indices], 
                attrs: {
                    'label': the label of this object
                }
        }
        'kfs': {
            kid: # keyframe index
                data = [[normalized image corners \in R^{4}, occlusion], ...]
                attrs: {
                    'seg2idx' [(kid, idx), ...] # the idx here is the list index in data
                }
        }
    },
    ...
}
# For estimated segments, the oid is the segment index instead of the object index.

```