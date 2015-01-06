package jgibblda;

import java.io.File;

public class Estimation {

    // output model
    protected Model trnModel;
    LDAOption option;
    int numstats;
    double[][] thetasum;
    double[][] phisum;

    public boolean init(LDAOption option){
        this.option = option;
        trnModel = new Model();

        if (option.est){
            if (!trnModel.initNewModel(option))
                return false;
            trnModel.data.localDict.writeWordMap(option.dir + File.separator + option.wordMapFileName);
        }
        else if (option.estc){
            if (!trnModel.initEstimatedModel(option))
                return false;
        }

        return true;
    }

    public void estimate() {

        int dispcol = 0;

        thetasum = new double[trnModel.M][trnModel.K];
        phisum = new double[trnModel.K][trnModel.V];

        System.out.println("Sampling " + 10000
                + " iterations with burn-in of " + 2000 + " (B/S="
                + 100 + ").");

        for (int i = 0; i < trnModel.niters; i++){
            // for all z_i
            for (int m = 0; m < trnModel.M; m++){
                for (int n = 0; n < trnModel.data.docs[m].length; n++){
                    // z_i = z[m][n]
                    // sample from p(z_i|z_-i, w)
                    int topic = sampling(m, n);
                    trnModel.z[m].set(n, topic);
                }// end for each word
            }// end for each document


            if ((i < 2000) && (i % 100 == 0)) {
                System.out.print("B");
                dispcol++;
            }
            // display progress
            if ((i >= 2000) && (i % 100 == 0)) {
                System.out.print("S");
                dispcol++;
            }
            // get statistics after burn-in
            if ((i >= 2000) && (i % 10 == 0)) {
                updateParams();
                System.out.print("|");
                if (i % 100 != 0)
                    dispcol++;
            }
            if (dispcol >= 100) {
                System.out.println();
                dispcol = 0;
            }
        }// end iterations

        for (int m = 0; m < trnModel.M; m++) {
            for (int k = 0; k < trnModel.K; k++) {
                trnModel.theta[m][k] = thetasum[m][k] / numstats;
            }
        }

        for (int k = 0; k < trnModel.K; k++) {
            for (int w = 0; w < trnModel.V; w++) {
                trnModel.phi[k][w] = phisum[k][w] / numstats;
            }
        }

        trnModel.saveModel("model-final");
    }

    /**
     * Do sampling
     * @param m document number
     * @param n word number
     * @return topic id
     */
    public int sampling(int m, int n){
        // remove z_i from the count variable
        int topic = trnModel.z[m].get(n);
        int w = trnModel.data.docs[m].words[n];

        trnModel.nw[w][topic] -= 1;
        trnModel.nd[m][topic] -= 1;
        trnModel.nwsum[topic] -= 1;
        trnModel.ndsum[m] -= 1;

        double Vbeta = trnModel.V * trnModel.beta;
        double Kalpha = trnModel.K * trnModel.alpha;

        //do multinominal sampling via cumulative method
        for (int k = 0; k < trnModel.K; k++){
            trnModel.p[k] = (trnModel.nw[w][k] + trnModel.beta)/(trnModel.nwsum[k] + Vbeta) *
                    (trnModel.nd[m][k] + trnModel.alpha)/(trnModel.ndsum[m] + Kalpha);
        }

        // cumulate multinomial parameters
        for (int k = 1; k < trnModel.K; k++){
            trnModel.p[k] += trnModel.p[k - 1];
        }

        // scaled sample because of unnormalized p[]
        double u = Math.random() * trnModel.p[trnModel.K - 1];

        for (topic = 0; topic < trnModel.K; topic++){
            if (trnModel.p[topic] > u) //sample topic w.r.t distribution p
                break;
        }

        // add newly estimated z_i to count variables
        trnModel.nw[w][topic] += 1;
        trnModel.nd[m][topic] += 1;
        trnModel.nwsum[topic] += 1;
        trnModel.ndsum[m] += 1;

        return topic;
    }

    public void computeTheta(){
        for (int m = 0; m < trnModel.M; m++){
            for (int k = 0; k < trnModel.K; k++){
                thetasum[m][k] += (trnModel.nd[m][k] + trnModel.alpha) / (trnModel.ndsum[m] + trnModel.K * trnModel.alpha);
            }
        }
    }

    public void computePhi(){
        for (int k = 0; k < trnModel.K; k++){
            for (int w = 0; w < trnModel.V; w++){
                phisum[k][w] += (trnModel.nw[w][k] + trnModel.beta) / (trnModel.nwsum[k] + trnModel.V * trnModel.beta);
            }
        }
    }

    public void updateParams() {

        computeTheta();
        computePhi();
        numstats++;
    }
}

