from abcpy.backend_mpi import BackendMPI

if __name__ == "__main__":

    backend = BackendMPI()
    data = list(range(100))

    def square(x):
        return x**2

    class staticfunctest:
        @staticmethod 
        def cube(x):
            return x**3


    datachunk_pds = backend.parallelize(data)
    print("Worker with Rank", backend.rank, "has", datachunk_pds.python_list)

    mapres_pds = backend.map(square, datachunk_pds)
    print ("Worker with Rank", backend.rank, "got map result", mapres_pds.python_list)

    print("Result of the map is:",backend.collect(mapres_pds))
    print("Original Data was:",backend.collect(datachunk_pds))

    mapres_pds = backend.map(staticfunctest.cube, datachunk_pds)
    print("Result of the map is:",backend.collect(mapres_pds))

    bcast_bds = backend.broadcast(data)
    #print("Broadcast at Rank", backend.rank, "has", backend.bds_store[bcast_bds.bds_id])

    for i in range(0, backend.size):
        print("Broadcasted data at Rank", i, "has", backend.bds_store[bcast_bds.bds_id])
