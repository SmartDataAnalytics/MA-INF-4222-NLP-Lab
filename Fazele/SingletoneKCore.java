package com.company;

import interviews.graphs.*;

public class SingletoneKCore {
    //private Graph graph;
    private  static interviews.graphs.KCore kc;
    //private  interviews.graphs.Graph g1;

    public interviews.graphs.KCore KC(Graph graph) {
        if (kc == (null))
        {
            kc = new interviews.graphs.KCore(graph.G1());
        }
        kc.computeWeighted();
        return kc;
    }

    public void setKC(Graph graph)
    {
        kc = new interviews.graphs.KCore(graph.G1());
    }

    public void renew_KC(){
        kc = null;
    }

}
