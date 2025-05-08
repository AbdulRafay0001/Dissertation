# test.py

def head(filepath, n=10):
    with open(filepath, 'r', encoding='latin-1') as f:
        for i in range(n):
            line = f.readline()
            if not line:
                break
            print(line.rstrip())

if __name__ == "__main__":
    head("ml-1m/ratings.dat", 10)

