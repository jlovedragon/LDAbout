package jgibblda;


public class LDA implements Runnable{

	@Override
	public void run() {
		LDAOption option = new LDAOption();
		
		option.dir = "/home/hadoop/Quentin_Hsu/sohuNews/";
		//option.dir = "/home/quentin/Downloads/Sample/test";
		option.dfile = "newdocs.dat";
		option.est = true;  /////
		///option.estc = true;
		option.inf = false;
		option.modelName = "model-final";
		option.niters = 10000;
		option.alpha = 2;
		option.K = 3;

		Estimation estimation = new Estimation();
		estimation.init(option);
		estimation.estimate();
		/*
		Estimator estimator = new Estimator();
		estimator.init(option);
		estimator.estimate();
		*/
	}

	public static void main(String[] args) {
		new LDA().run();
	}
	
}