from ocelotml import predict_from_file, load_models
import time

def main():
    model =  load_models('hh')
    start = time.time()
    predict_from_file("./dimer.xyz", model=model)[0]
    end = time.time()
    print(end-start)

if __name__ == "__main__":
    main()