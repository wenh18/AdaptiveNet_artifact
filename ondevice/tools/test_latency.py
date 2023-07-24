import numpy as np
import torch
import time
import torchvision
import os
import sys

def get_latency(model, batchsize, test_times=200):
    model.eval()
    model.cuda()
    lats = []
    with torch.no_grad():
        for i in range(test_times):
            x = torch.rand(batchsize, 3, 224, 224).cuda()
            t1 = time.time()
            _ = model(x)
            torch.cuda.synchronize()
            t2 = time.time() - t1
            if i > 100:
                lats.append(t2)
    return np.mean(lats)


def main():
    batch_size = int(sys.argv[1])
    model_name = sys.argv[2]
    if model_name == "resnet50":
        model = torchvision.models.resnet50()
    else :
        model = torchvision.models.mobilenet_v2()
        
    lat = get_latency(model,batch_size)
    print("process_id : {} model_name : {} latency:{:.2f}ms batch_size:{}".format(os.getpid(), model_name, lat * 1000, batch_size))
    model =  torchvision.models.mobilenet_v2()

if __name__ == '__main__':
    main()
