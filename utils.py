import torch


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def Calculate_mAP(tst_label, tst_binary, db_label, db_binary, top_k=-1):
    if top_k == -1:
        top_k = db_binary.size(0)
    mAP = 0.
    # 检索项的个数
    num_query = tst_binary.size(0)
    NS = (torch.arange(top_k) + 1).float()
    for idx in range(num_query):
        query_label = tst_label[idx].unsqueeze(0)
        query_binary = tst_binary[idx]
        result = query_label.mm(db_label.t()).squeeze(0) > 0
        # 数据库中图片按照与查询项的相似性排序
        hamm = calc_hammingDist(query_binary, db_binary)
        _, index = hamm.sort()
        index.squeeze_()
        result = result[index[: top_k]].float()
        tsum = torch.sum(result)
        if tsum == 0:
            continue
        # 数据库中与查询项有相同标签的图片数
        all_similar = torch.sum(result)
        # 注：准确率和召回率可由此计算
        accuracy = torch.cumsum(result, dim=0) / torch.sum(result)
        mAP += float(torch.sum(result * accuracy / NS).item())
    return mAP/num_query


def Logtrick(x, use_gpu):
    if use_gpu:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]).cuda())
    else:
        lt = torch.log(1+torch.exp(-torch.abs(x))) + torch.max(x, torch.FloatTensor([0.]))
    return lt


def generate_hash_code(dataloader, img_net, txt_net, use_gpu):
    bs, tags, clses = [], [], []
    for img, tag, label, _ in dataloader:
        clses.append(label)
        if use_gpu:
            img, tag = img.cuda(), tag.cuda()
        img = img_net(img).cpu().data
        bs.append(img)

        tag = tag.unsqueeze(1).unsqueeze(-1).type(torch.float)
        tag = txt_net(tag).cpu().data
        tags.append(tag)
    return torch.sign(torch.cat(bs)), torch.sign(torch.cat(tags)), torch.cat(clses)
