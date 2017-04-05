from abcpy.backends import BackendMPI

def square(x):
    return x**2


if __name__ == "__main__":

    backend = BackendMPI()
    data = list(range(100))

    datachunk_pds = backend.parallelize(data)
    print("Worker with Rank", backend.rank, "has", datachunk_pds.python_list)

    mapres_pds = backend.map(square, datachunk_pds)
    print ("Worker with Rank", backend.rank, "got map result", mapres_pds.python_list)

    print("Result of the map is:",backend.collect(mapres_pds))
    print("Original Data was:",backend.collect(datachunk_pds))
