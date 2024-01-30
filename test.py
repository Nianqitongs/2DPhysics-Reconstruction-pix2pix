# 院校：南京信息工程大学
# 院系：自动化学院
# 开发时间：2024/1/30 8:54
import torch
import time
import matplotlib.pyplot as plt
def test_per_epoch(model,data_loader,epoch,args,criterionL1):
    model.eval()
    with torch.no_grad():
        data_loader_iter = iter(data_loader)
        test_data,test_condition,test_label = next(data_loader_iter)
        test_data,test_condition,test_label = test_data.to(args.device),test_condition.to(args.device),test_label.to(args.device)
        # if epoch == 1:
        #     plt.figure(figsize=(8, 8))
        #     for i in range(8):
        #         for j in range(8):
        #             plt.subplot(8, 8, i * 8 + j + 1)
        #             plt.imshow(test_label[i * 8 + j, 0, :, :].cpu().numpy(), cmap='gray')
        #             plt.axis(False)
        #     plt.savefig(args.label_path + "label.png")
        #     plt.tight_layout()
        #     plt.close()

        condition_input = test_condition.view(test_data.size(0),2,1,1).to(args.device)
        real_input = torch.cat([test_data, torch.ones_like(test_data) * condition_input], dim=1).to(args.device)
        time_start = time.time()
        predicate = model(real_input)
        time_end = time.time()
        time_sum = time_end - time_start
        print(time_sum)
        #predicate = model(test_data,test_condition)
        # plt.figure(figsize=(8, 8))
        # for i in range(8):
        #     for j in range(8):
        #         plt.subplot(8, 8, i * 8 + j + 1)
        #         plt.imshow(predicate[i * 8 + j, 0, :, :].cpu().numpy(), cmap='gray')
        #         plt.axis(False)
        # plt.savefig(args.predicate_path + f'epoch:{epoch+1}.png')
        # plt.tight_layout()
        # plt.close()
    model.train()
    return criterionL1(predicate,test_label)