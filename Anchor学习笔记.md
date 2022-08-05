# Anchor学习笔记

## 生成多个anchor

输入：图像，缩放比，宽高比

输出：anchors

原理：以每个像素中心点生成w*h*(n+m-1)个anchor，有不同的缩放比s（大小），和宽高比r（形状）。

代码：

```python
def multibox_prior(data, sizes, ratios):
    in_height, in_width = data.shape[-2:]
    #torch.shape 返回输入tensor张量的维度大小 data.tensor[1,3,561,728] data.shape[-2:]取从右数两个
    device,num_size,num_ratios=data.device,len(sizes),len(ratios)
    #得到num_size=n,num_ratios=m
    boxes_per_pixel=(num_size+num_ratios-1)
    #每个像素点的anchor数n+m-1
    size_tensor = torch.tensor(sizes, device=device)
    ratio_tensor = torch.tensor(ratios, device=device)
    #转到gpu,list转tensor
    
    # 为了将锚点移动到像素的中心，需要设置偏移量。
    # 因为一个像素的的高为1且宽为1，我们选择偏移我们的中心0.5
    offset_h, offset_w = 0.5, 0.5
    steps_h = 1.0 / in_height  # 在y轴上缩放步长，steps_h=1/100=0.01
    steps_w = 1.0 / in_width  # 在x轴上缩放步长，steps_w=1/80=0.0125
    
    
    # 生成锚框的所有中心点 561*728个点
    center_h = (torch.arange(in_height, device=device)+offset_h)*steps_h
    center_w = (torch.arange(in_width, device=device)+offset_w)*steps_w
    #z=torch.arange(1,6) z tensor([1, 2, 3, 4, 5]), 每个像素点向里右下偏移0.5*0.5，再×缩放步长 center_h 561个点， center_w 782个点
    shift_y, shift_x = torch.meshgrid(center_h, center_w)
    #  torch.meshgrid（）的功能是生成网格，可以用于生成坐标。
    # center_h： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    # center_w： tensor([0.1250, 0.3750, 0.6250, 0.8750])
    #  shift_y tensor([
    #               [0.1250, 0.1250, 0.1250, 0.1250],
    #               [0.3750, 0.3750, 0.3750, 0.3750],
    #               [0.6250, 0.6250, 0.6250, 0.6250],
    #               [0.8750, 0.8750, 0.8750, 0.8750]]) 
    #  shift_y shift_x 都是561*728=408408个点，shift_y提供每个点行位置， shift_x提供每个点列位置 
        
    # shift_x tensor([
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750],
    #               [0.1250, 0.3750, 0.6250, 0.8750]])
    shift_y, shift_x = shift_y.reshape(-1), shift_x.reshape(-1) #展平
    # 全部每个像素中心点坐标
    # tensor([0.1250, 0.1250, 0.1250, 0.1250, 0.3750, 0.3750, 0.3750, 0.3750, 0.6250,
    #     0.6250, 0.6250, 0.6250, 0.8750, 0.8750, 0.8750, 0.8750]) 
    # tensor([0.1250, 0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750, 0.1250,
    #     0.3750, 0.6250, 0.8750, 0.1250, 0.3750, 0.6250, 0.8750])
    
    
    # 为每个像素点生成“boxes_per_pixel”个锚框的高和宽
    # 之后用于创建锚框的四角坐标(xmin,xmax,ymin,ymax)
    w = torch.cat((size_tensor * torch.sqrt(ratio_tensor[0]),
                   sizes[0] * torch.sqrt(ratio_tensor[1:])))* in_height / in_width  # 处理矩形输入
    h = torch.cat((size_tensor / torch.sqrt(ratio_tensor[0]),
                   sizes[0] / torch.sqrt(ratio_tensor[1:])))
    #只包含s1和r1,共n+m-1个，*in_height / in_width应该是调整尺寸显示用的
    # 除以2来获得半高和半宽
    anchor_manipulations = torch.stack((-w, -h, w, h)).T.repeat(
                                        in_height * in_width, 1) / 2
    #原来5个高宽，重复408408次，再除2
    
    # 每个中心点都将有“boxes_per_pixel”个锚框，
    # 所以生成含所有锚框中心的网格，重复了“boxes_per_pixel”次
    out_grid = torch.stack([shift_x, shift_y, shift_x, shift_y],
                dim=1).repeat_interleave(boxes_per_pixel, dim=0)
    output = out_grid + anchor_manipulations
    return output.unsqueeze(0)
	#.unsqueeze(0)，增加第一个维度
```

## IOU计算

输入：anchor1，anchor2

输出：列为box1，行为box2序列的iou表

原理：交集/并集

```python
def box_iou(boxes1, boxes2):
    #boxes1(左上角x，左上角y，右下角x，右下角y)
#这边举个例子：boxes1 [1,1,3,3],[0,0,2,4],[1,2,3,4]]   ；boxes2[[0,0,3,3],[2,0,5,2]]
    # 首先计算一个框的面积（长X宽）
    box_area = lambda boxes: ((boxes[:, 2] - boxes[:, 0])*(boxes[:, 3] - boxes[:, 1]))
    #定义匿名函数box_area boxes为输入，计算面积
    areas1 = box_area(boxes1)
    areas2 = box_area(boxes2)
    
    # inter_upperlefts,inter_lowerrights,inters的形状:
    # (boxes1的数量,boxes2的数量,2)
    inter_upperlefts = torch.max(boxes1[:, None, :2], boxes2[:, :2])
    #boxes1：tensor([[1, 1, 3, 3],
    #               [0, 0, 2, 4],
    #               [1, 2, 3, 4]])
    #boxes2:tensor([[0, 0, 3, 3],
    #               [2, 0, 5, 2]])
    #输入是二维，所以两个：，None增加一个维度，#由于它们的维度不同，所以要用广播机制，真正计算的时候，是下面这样的
    #tensor([[[1, 1],[1, 1]],[[0, 0],[0, 0]],[[1, 2],[1, 2]]])
    #tensor([[0, 0],[2, 0]],[[0, 0],[2, 0]], [[0, 0],[2, 0]])
    #此时inter_upperlefts 为：
    #tensor([[[1, 1],
    #         [2, 1]],
    #        [[0, 0],
    #        [2, 0]],
    #       [[1, 2],
    #        [2, 2]]])
    #由于维度不同，所以max后输出的是坐标
    inter_lowerrights  = torch.min(boxes1[:, None, 2:], boxes[:, 2:])
    #左侧取max，右侧取min，画图可知是交集
    
    inters = (inter_lowerrights - inter_upperlefts).clamp(min=0)
    #clamp(min=0)用来限制inters最小不能低于0
    #tensor([[[0.10, 0.20],[0.00, 0.10]]])
    inter_areas = inters[:, :, 0]* inters[:, :, 1]
    union_areas = areas1[:, None] + areas2 - inter_areas
    #横坐标对应gt，纵坐标对应anchor，直接对应下面分配的表格
    return inter_areas / union_areas
```

## 将ground truth 分配给 anchor

输入：gt，anchor，device，阈值

输出：anchors对应的类别

原理：给定图像，假设锚框是$A_1, A_2, \ldots, A_{n_a}$，真实边界框是$B_1, B_2, \ldots, B_{n_b}$，其中$n_a \geq n_b$。
让我们定义一个矩阵$\mathbf{X} \in \mathbb{R}^{n_a \times n_b}$，其中第$i$行、第$j$列的元素$x_{ij}$是锚框$A_i$和真实边界框$B_j$的IoU。
该算法包含以下步骤：

1. 在矩阵$\mathbf{X}$中找到最大的元素，并将它的行索引和列索引分别表示为$i_1$和$j_1$。然后将真实边界框$B_{j_1}$分配给锚框$A_{i_1}$。这很直观，因为$A_{i_1}$和$B_{j_1}$是所有锚框和真实边界框配对中最相近的。在第一个分配完成后，丢弃矩阵中${i_1}^\mathrm{th}$行和${j_1}^\mathrm{th}$列中的所有元素。

1. 在矩阵$\mathbf{X}$中找到剩余元素中最大的元素，并将它的行索引和列索引分别表示为$i_2$和$j_2$。我们将真实边界框$B_{j_2}$分配给锚框$A_{i_2}$，并丢弃矩阵中${i_2}^\mathrm{th}$行和${j_2}^\mathrm{th}$列中的所有元素。

1. 此时，矩阵$\mathbf{X}$中两行和两列中的元素已被丢弃。我们继续，直到丢弃掉矩阵$\mathbf{X}$中$n_b$列中的所有元素。此时，我们已经为这$n_b$个锚框各自分配了一个真实边界框。

1. 只遍历剩下的$n_a - n_b$个锚框。例如，给定任何锚框$A_i$，在矩阵$\mathbf{X}$的第$i^\mathrm{th}$行中找到与$A_i$的IoU最大的真实边界框$B_j$，只有当此IoU大于预定义的阈值时，才将$B_j$分配给$A_i$。

   <img src="C:\Users\11790\AppData\Roaming\Typora\typora-user-images\image-20220804163850728.png" alt="image-20220804163850728" style="zoom:50%;" />

```python
def assign_anchor_to_bbox(ground_truth, anchors, device, iou_threshold=0.5):
    num_anchors, num_gt_boxes = anchors.shape[0], ground_truth.shape[0]
    #分配anchor个数和gt个数
    jaccard = box_iou(anchors, ground_truth)
    #对应上图iou_map
      """
    tensor([[0.0536, 0.0000],
            [0.1417, 0.0000],
            [0.0000, 0.5657],
            [0.0000, 0.2059],
            [0.0000, 0.7459]])
    """
    anchors_bbox_map = torch.full((num_anchors,), -1, dtype=torch.long, device=device)
    #tensor([-1, -1, -1, -1, -1])
    #定义anchors_bbox_map来记录anchor分别对应着什么gt，anchors_bbox_map存放标签初始全为-1,为一维长度为num_anchors的tensor. torch.full用来填满-1
    jaccard_cp = jaccard.clone()
    # 先为每个bb分配一个anchor
    col_discard = torch.full((num_anchors,), -1)
    row_discard = torch.full((num_gt_boxes,), -1)
    #设置行列数分别等于anchor和gt数的全为-1，用来丢弃其余元素
    for _ in range(num_gt_boxes):
        max_idx = torch.argmax(jaccard_cp)
        #输出最大iou的序号，循环次数为gt个数，例子中有两个gt，第一次0.7459输出9，后面去除行列，第二次0.1417输出2
        box_idx = (max_idx % num_gt_boxes).long()
        #列索引，取余数，9%2=1，既对应列索引，又对应gt号
        anc_idx = (max_idx / num_gt_boxes).long()
        #行索引，取商，9/2=4，所以索引到(4,1)=0.7459，既对应行索引，又对应anchor号
        anchors_bbox_map[anc_idx] = box_idx
        #anchor4对应gt=1，把gt分配给anchor
        jaccard_cp[:, box_idx] = col_discard
        jaccard_cp[anc_idx, :] = row_discard
        #丢弃最大元素所在行的其余元素，将其设为-1
        
        # 遍历剩余的na−nb个锚框
        # 处理还未被分配的anchor, 要求满足iou_threshold
        for i in range(num_anchors):
            # 索引等于初始值-1 的就是剩下的锚框
            if anchors_bbox_map[i] == -1:
                j = torch.argmax(jaccard[i, :])
                # 根据阈值，决定是否分配真实边界框
            if jaccard[i, j] >= iou_threshold:
                anchors_bbox_map[i] = j
        # 每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
    return anchors_bbox_map
```

## 用ground truth标记anchor

输入：anchors, gt

输出：bbox_offset, bbox_mask, class_labels

原理：标记anchor对应的类别，输出三个标签，类别，掩码，偏移量

```python
‘’‘
labels = torch.tensor([[0, 0.1, 0.08, 0.52, 0.92],
                         [1, 0.55, 0.2, 0.9, 0.88]])
anchors = torch.tensor([[0, 0.1, 0.2, 0.3], [0.15, 0.2, 0.4, 0.4],
                    [0.63, 0.05, 0.88, 0.98], [0.66, 0.45, 0.8, 0.8],
                    [0.57, 0.3, 0.92, 0.9]])
‘’‘
# anchors输入的锚框[1,锚框总数，4] labels真实标签[bn,真实锚框数，5]
#在下面的例子中为5，gt数目为2
def multibox_target(anchors, labels):
    """使用真实边界框标记锚框"""
    batch_size, anchors = labels.shape[0], anchors.squeeze(0)
    #.squeeze(0)作用为降维
    batch_offset, batch_mask, batch_class_labels = [], [], []
    device, num_anchors = anchors.device, anchors.shape[0]
    
    for i in range(batch_size):
        label = labels[i, :, :]
        #取出每个batch的gt
        anchors_bbox_map = assign_anchor_to_bbox(label[:, 1:], anchors, device)
        #得到每个anchor对应当前gt的索引
        # assign_anchor_to_bbox函数返回，每个anchor分配的真实bb对应的索引, 若未分配任何bb则为-1
        # tensor([-1,  0,  1, -1,  1])
        #这边label[:, 1:] 从1开始是因为，求IOU的时候不需要用到类别
        bbox_mask = ((anchors_bbox_map >= 0).float().unsqueeze(-1).repeat(1,4))
        # bbox_mask: (锚框总数, 4), 0代表背景, 1代表非背景
        #.unsqueeze(-1)作用为升维
        #没有repeat之前，当参数只有两个时：（行的重复倍数，列的重复倍数）
        #tensor([[0.],
        #        [1.],
        #        [1.],
        #        [0.],
        #        [1.]])
        #repeat之后
        #tensor([[0., 0., 0., 0.],
      #          [1., 1., 1., 1.],
      #          [1., 1., 1., 1.],
      #          [0., 0., 0., 0.],
      #          [1., 1., 1., 1.]])
        class_labels = torch.zeros(num_anchors, dtype = torch.long, device=device)
        assigned_bb = torch.zeros((num_anchors, 4), dtype = torch.float32, device=device)
        #将类标签和分配的边界框坐标初始化为零，tensor([0, 0, 0, 0, 0])
        indices_true = torch.nonzero(anchors_bbox_map >= 0)
        #(anchors_bbox_map >= 0)返回0,1,1,0,1，torch.nonzero，返回非背景序号1,2,4
        bb_idx = anchors_bbox_map[indices_true]
        #返回0,1,1，非背景序号对应的gt类别，gt=0对应label第一个类别为0，gt=1对应label第一个类别为1，为下一行提供索引
        class_labels[indices_true] = label[bb_idx, 0].long() + 1
        # 背景为0，新类的整数索引递增1，label[bb_idx, 0]选择label的第一列0/1
        #class_lable为[0, 1, 2, 0, 2]
        assigned_bb[indices_true] = label[bb_idx, 1:]
        #把真实标注好的边界框的坐标值赋给与其对应的某一锚框
        offset = offset_boxes(anchors, assigned_bb) * bbox_mask
        # 偏移量转换，bbox_mask过滤掉背景
        batch_offset.append(offset.reshape(-1))
        batch_mask.append(bbox_mask.reshape(-1))
        batch_class_labels.append(class_labels)
    bbox_offset = torch.stack(batch_offset)
    bbox_mask = torch.stack(batch_mask)
    class_labels = torch.stack(batch_class_labels)
    return (bbox_offset, bbox_mask, class_labels)
 
```

## 非极大值抑制NMS

输入：boundingbox，置信度，阈值

输出：唯一boundingbox

原理：当有许多锚框时，可能会输出许多相似的具有明显重叠的预测边界框，都围绕着同一目标。
为了简化输出，我们可以使用*非极大值抑制*（non-maximum suppression，NMS）合并属于同一目标的类似的预测边界框。

以下是非极大值抑制的工作原理。
对于一个预测边界框$B$，目标检测模型会计算每个类别的预测概率。
假设最大的预测概率为$p$，则该概率所对应的类别$B$即为预测的类别。
具体来说，我们将$p$称为预测边界框$B$的*置信度*（confidence）。
在同一张图像中，所有预测的非背景边界框都按置信度降序排序，以生成列表$L$。然后我们通过以下步骤操作排序列表$L$：

1. 从$L$中选取置信度最高的预测边界框$B_1$作为基准，然后将所有与$B_1$的IoU超过预定阈值$\epsilon$的非基准预测边界框从$L$中移除。这时，$L$保留了置信度最高的预测边界框，去除了与其太过相似的其他预测边界框。简而言之，那些具有*非极大值*置信度的边界框被*抑制*了。

1. 从$L$中选取置信度第二高的预测边界框$B_2$作为又一个基准，然后将所有与$B_2$的IoU大于$\epsilon$的非基准预测边界框从$L$中移除。

1. 重复上述过程，直到$L$中的所有预测边界框都曾被用作基准。此时，$L$中任意一对预测边界框的IoU都小于阈值$\epsilon$；因此，没有一对边界框过于相似。

1. 输出列表$L$中的所有预测边界框。

   ```python
   # 按降序对置信度进行排序并返回其索引
   def nms(bboxs, scores, threshold):
       order = torch.argsort(scores, dim=-1, descending=True) #[0,3,1,2]
       # 取出分数从大到小排列的索引 order为排序后的得分对应的原数组索引值
       # torch.argsort返回排序后的值所对应原a的下标，即torch.sort()返回的indices
       keep = []
       # 这边的keep用于存放，NMS后剩余的方框(保存所有结果框的索引值)
       while order.numel() > 0:
           # .numel()获取tensor中一共包含多少个元素
           if order.numel() == 1: #只剩下一个时候直接放进去，例子中只剩下3
               keep.append(order.item())
               break
           else:
               i = order[0].item()
               # 置信度最高的索引
               keep.append(i) 
               # keep保留的是索引值，不是具体的分数。
           iou = box_iou(bboxs[i, :].reshape(-1, 4),
                         bboxs[order[1:], :].reshape(-1, 4)).reshape(-1)
            # bboxs[i, :]取置信度最大框，bboxs[order[1:], :]取剩下的框，reshape拉到一维
            # 添加本次置信度最高的boundingbox的index；在上面的例子中，第一次加入‘0’
            # 计算最大得分的bboxs[i]与其余各框的IOU
            #第一次，iou为tensor([0.0000, 0.7368, 0.5454])
           idx = torch.nonzero((iou <= threshold)).reshape(-1)  
           # 返回非零元素的索引，操作类似于之前
           # 保留iou小于阈值的剩余bboxs,iou小表示两个box交集少，可能是另一个物体的框，故需要保留
           
           # 待处理boundingbox的个数为0时，结束循环
           if idx.numel() == 0:
               break
           # 把留下来框在进行NMS操作
           # 这边留下的框是去除当前操作的框，和当前操作的框重叠度大于threshold的框
           # 因为处理的时候是对tensor([0.0000, 0.7368, 0.5454])进行处理的，去掉了第一个最大的框子
           #最后返回的时候，要把第一个位置给他加上去。也就是idx+1后挪一位
           order = order[idx + 1] #iou小于阈值的框
       return torch.tensor(keep,device=bboxs.device)
   ```

   ## 用非极大值抑制应用于预测边界框

   原理：在已经将gt和anchor对应过后，分出背景和非背景，结合置信度得出最符合的bounding box

   ```python
   #cls_probs = torch.tensor([[0] * 4,  # 背景的预测概率
   #    [0.9, 0.8, 0.7, 0.1],  # 狗的预测概率
   #    [0.1, 0.2, 0.3, 0.9]])  # 猫的预测概率
   #anchors = torch.tensor([[0.1, 0.08, 0.52, 0.92], [0.08, 0.2, 0.56, 0.95],
   # [0.15, 0.3, 0.62, 0.91], [0.55, 0.2, 0.9, 0.88]])
   #offset_preds = torch.tensor([0] * anchors.numel()).reshape(-1, 4)  #这边为了方便 偏移量都设为0
   def multibox_detection(cls_probs, offset_preds, anchors, nms_threshold=0.5,
                          pos_threshold=0.0099):
       """使用非极大值抑制来预测边界框。"""
       device, batch_size = cls_probs.device, cls_probs.shape[0]
       #batch_size = 1 
       anchors = anchors.squeeze(0)
       # 保存最终的输出
       out = []
       for i in range(batch_size):
           # 预测概率和预测的偏移量
           cls_prob, offset_pred = cls_probs[i], offset_preds[i].reshape(-1, 4)
   
           # 非背景的概率及其类别索引
           #torch.max(input, dim)，dim=0代表每列的最大值
           #函数会返回两个tensor，第一个tensor是每行的最大值；第二个tensor是每行最大值的索引。
           conf, class_id = torch.max(cls_prob[1:], 0)
           #tensor([0.90, 0.80, 0.70, 0.90]) tensor([0, 0, 0, 1])
   
           # 预测的边界框坐标
           predicted_bb = offset_inverse(anchors, offset_pred)
           # 对置信度进行排序并返回其索引[0,3,1,2]
           all_id_sorted = torch.argsort(conf, dim=-1, descending=True)
   
           keep = nms(predicted_bb, conf, nms_threshold)  # 非极大值抑制结果 [0,3]
           # 找到所有的 non_keep 索引，并将类设置为背景
           non_keep = []
           for i in range(all_id_sorted.numel()):
               res = all_id_sorted[i] in keep
               if not res:
                   non_keep.append(all_id_sorted[i].item())
           non_keep = torch.tensor(non_keep) # [1,2]
           # 将类设置为背景-1
           class_id[non_keep] = -1 # tensor([ 0, -1, -1,  1])
           # 对应的类别标签
           class_id = class_id[all_id_sorted] #tensor([ 0,  1, -1, -1])
           # 排序,conf为tensor([0.90, 0.90, 0.80, 0.70])
           conf, predicted_bb = conf[all_id_sorted], predicted_bb[all_id_sorted]
   
           # `pos_threshold` 是一个用于非背景预测的阈值
           below_min_idx = (conf < pos_threshold)
           class_id[below_min_idx] = -1
           conf[below_min_idx] = 1 - conf[below_min_idx]
   
           pred_info = torch.cat(
               (class_id.unsqueeze(1), conf.unsqueeze(1), predicted_bb), dim=1)
           out.append(pred_info)
       return torch.stack(out)
   ```

   <img src="C:\Users\11790\AppData\Roaming\Typora\typora-user-images\image-20220805163215640.png" alt="image-20220805163215640"  />

维度改变很关键

output.shape = torch.Size([1, 4, 6])
