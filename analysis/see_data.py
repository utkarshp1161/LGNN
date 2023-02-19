import pickle
import pdb as pdb 


def main():
    #pdb.set_trace()
    config = pickle.load(open("/home/sire/phd/srz228573/scratch/benchmarking/LGNN/results/9-Spring-data/0/initial-configs_0.pkl", "rb"))
    model_states = pickle.load(open("/home/sire/phd/srz228573/scratch/benchmarking/LGNN/results/9-Spring-data/0/model_states_0.pkl", "rb"))
    print(model_states)


if __name__ == "__main__":
    main()