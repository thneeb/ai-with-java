package de.neebs.ai.control.dl4j;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.api.OptimizationAlgorithm;
import org.deeplearning4j.nn.conf.BackpropType;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.distribution.UniformDistribution;
import org.deeplearning4j.nn.conf.layers.DenseLayer;
import org.deeplearning4j.nn.conf.layers.OutputLayer;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.learning.config.Sgd;
import org.nd4j.linalg.lossfunctions.LossFunctions;
import org.springframework.stereotype.Service;

@Service
@Slf4j
public class FirstExample {
    public void execute() {
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.1))
                .miniBatch(false)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(2)
                        .activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(1)
                        .activation(Activation.SIGMOID)
                        .build())
                .build();
/*
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .weightInit(WeightInit.XAVIER)
                .miniBatch(false)
//                .activation(Activation.SIGMOID)
//                .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
                .updater(new Sgd(0.05))
                // ... other hyperparameters
                .list()
//                .layer(new DenseLayer.Builder().nIn(2).nOut(2).build())
                .layer(new DenseLayer.Builder().nIn(2).nOut(2).activation(Activation.SIGMOID)
                        .build())
                .layer(new OutputLayer.Builder().nIn(2).nOut(1).lossFunction(LossFunctions.LossFunction.MSE)
                        .activation(Activation.SIGMOID)
                        .build())
                .backpropType(BackpropType.Standard)
                .build();
*/
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();
        //record score once every 100 iterations
        model.setListeners(new ScoreIterationListener(100));

        // 0, 0 => 0
        // 0, 1 => 1
        // 1, 0 => 1
        // 1, 1 => 0
        INDArray input = Nd4j.zeros(4, 2);
        input.putScalar(new int[] {0, 0}, 0);
        input.putScalar(new int[] {0, 1}, 0);
        input.putScalar(new int[] {1, 0}, 0);
        input.putScalar(new int[] {1, 1}, 1);
        input.putScalar(new int[] {2, 0}, 1);
        input.putScalar(new int[] {2, 1}, 0);
        input.putScalar(new int[] {3, 0}, 1);
        input.putScalar(new int[] {3, 1}, 1);

        INDArray labels  = Nd4j.zeros(4, 1);
        labels.putScalar(new int[] {0}, 0);
        labels.putScalar(new int[] {1}, 1);
        labels.putScalar(new int[] {2}, 1);
        labels.putScalar(new int[] {3}, 0);
        DataSet dataSet = new DataSet(input, labels);
        for (int i = 0; i < 1; i++) {
            model.fit(dataSet);
        }

        // create output for every training sample
        INDArray output = model.output(dataSet.getFeatures());
        System.out.println(output);

        // let Evaluation prints stats how often the right output had the highest value
        Evaluation eval = new Evaluation();
        eval.eval(dataSet.getLabels(), output);
        System.out.println(eval.stats());
    }

    public void execute2() {

        int seed = 1234;        // number used to initialize a pseudorandom number generator.
        int nEpochs = 1;    // number of training epochs

        log.info("Data preparation...");

        // list off input values, 4 training samples with data for 2
        // input-neurons each
        INDArray input = Nd4j.zeros(4, 2);

        // correspondending list with expected output values, 4 training samples
        // with data for 2 output-neurons each
        INDArray labels = Nd4j.zeros(4, 2);

        // create first dataset
        // when first input=0 and second input=0
        input.putScalar(new int[]{0, 0}, 0);
        input.putScalar(new int[]{0, 1}, 0);
        // then the first output fires for false, and the second is 0 (see class comment)
        labels.putScalar(new int[]{0, 0}, 1);
        labels.putScalar(new int[]{0, 1}, 0);

        // when first input=1 and second input=0
        input.putScalar(new int[]{1, 0}, 1);
        input.putScalar(new int[]{1, 1}, 0);
        // then xor is true, therefore the second output neuron fires
        labels.putScalar(new int[]{1, 0}, 0);
        labels.putScalar(new int[]{1, 1}, 1);

        // same as above
        input.putScalar(new int[]{2, 0}, 0);
        input.putScalar(new int[]{2, 1}, 1);
        labels.putScalar(new int[]{2, 0}, 0);
        labels.putScalar(new int[]{2, 1}, 1);

        // when both inputs fire, xor is false again - the first output should fire
        input.putScalar(new int[]{3, 0}, 1);
        input.putScalar(new int[]{3, 1}, 1);
        labels.putScalar(new int[]{3, 0}, 1);
        labels.putScalar(new int[]{3, 1}, 0);

        // create dataset object
        DataSet ds = new DataSet(input, labels);

        log.info("Network configuration and training...");

        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .updater(new Sgd(0.1))
                .seed(seed)
                .biasInit(0) // init the bias with 0 - empirical value, too
                // The networks can process the input more quickly and more accurately by ingesting
                // minibatches 5-10 elements at a time in parallel.
                // This example runs better without, because the dataset is smaller than the mini batch size
                .miniBatch(false)
                .list()
                .layer(new DenseLayer.Builder()
                        .nIn(2)
                        .nOut(4)
                        .activation(Activation.SIGMOID)
                        // random initialize weights with values between 0 and 1
                        .weightInit(new UniformDistribution(0, 1))
                        .build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.MSE)
                        .nOut(2)
                        .activation(Activation.SIGMOID)
                        .weightInit(new UniformDistribution(0, 1))
                        .build())
                .build();

        MultiLayerNetwork net = new MultiLayerNetwork(conf);
        net.init();

        // add an listener which outputs the error every 100 parameter updates
        net.setListeners(new ScoreIterationListener(100));

        // C&P from LSTMCharModellingExample
        // Print the number of parameters in the network (and for each layer)
        System.out.println(net.summary());

        // here the actual learning takes place
        for( int i=0; i < nEpochs; i++ ) {
            net.fit(ds);
        }

        // create output for every training sample
        INDArray output = net.output(ds.getFeatures());
        System.out.println(output);

        // let Evaluation prints stats how often the right output had the highest value
        Evaluation eval = new Evaluation();
        eval.eval(ds.getLabels(), output);
        System.out.println(eval.stats());
    }
}
