#TODO: WORK IN PROGRESS.

class ShortestPathDAGOracle(GraphOracle):
    def __init__(self, G, source, target):
        super().__init__(G)
        self.Z = None
        self.source = source
        self.target = target
    
    def _z_to_path(self, z):
        path = [self.source]
        for i in z:
            if i==0:
                pass
            elif i==1:
                a,b,_ = self.edgelist[i]
                if a==path[-1]:
                    path.append(b)
                elif b == path[-1]:
                    path.append(a)
        return path

    def max(self, v):
        '''
        Given the set of weights -v, returns the shortest path between source and target
        '''
        self.max_calls += 1
        self._weightG(-v) #CRITICAL: flips the weights so you can use the shortest-path
        path = nx.bellman_ford_path(self.G,source = self.source, target = self.target,weight='weight')
#         paths = nx.johnson(self.G,weight='weight')
#         path = paths[self.source][self.target]
        z = np.array(self._path_to_z(path))
        logging.debug('max shortest path: {}'.format(z))

        return np.inner(v, z), z
    
    def _makeZ(self):
        self.Z = np.array([self._path_to_z(path) for path in nx.all_simple_paths(self.G,self.source,self.target)])
    
    def _gfracmax(self, z0, l, shift, thetak, eta):
        '''
        For debugging purposes. Computes
            max_z (z0-z) A(l)^{-1/2} eta/(shift + thetak(z0-z)
        by explicitly enumerating all paths as Z's.
        '''
        all_paths = nx.all_simple_paths(self.G, self.source, self.target)
        Z = []
        paths = []
        for i,path in enumerate(all_paths):
            paths.append(path)
            z = self._path_to_z(path)
            Z.append(z)
        Z = np.array(Z)
        self.Z = Z
        
        def _gfrac(z):
            return np.sum((z0-z)*np.sqrt(1/l)*eta)/(shift + np.inner((z0-z),thetak) ) 
        idx = np.argmax([_gfrac(Z[i,:]) for i in range(Z.shape[0])])
        fracvalue = _gfrac(Z[idx,:])
        diffvalue = (Z[idx,:]@(-np.sqrt(1/l)*eta + fracvalue*thetak) 
                     - fracvalue*(shift+np.inner(thetak, z0)) + np.sum(z0*np.sqrt(1./l)*eta))
        return fracvalue, diffvalue 