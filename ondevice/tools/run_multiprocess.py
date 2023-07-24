from subprocess import call
import sys
from multiprocessing import Process

def run(name, modele_name, batch_size):
    print('process %s run'% name)
    call(['python', 'test_latency.py', batch_size, modele_name])
    
def main():
    batch_size = sys.argv[1]
    model_name = sys.argv[2]
    process_count = sys.argv[3]
    print("Process count : {} model_name : {} batch_size : {}".format(process_count, model_name, batch_size))
    for m in range(int(process_count)):
        p1=Process(target=run,args=(str(m), model_name, batch_size))
        p1.start()
    


if __name__ == '__main__':
    main()
