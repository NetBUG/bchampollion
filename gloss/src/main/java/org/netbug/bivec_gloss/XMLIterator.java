package org.netbug.bivec_gloss;
import org.apache.commons.io.FileUtils;
import org.apache.commons.io.FilenameUtils;
import org.apache.commons.lang3.tuple.Pair;
import org.deeplearning4j.datasets.iterator.DataSetIterator;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.dataset.api.DataSetPreProcessor;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.NoSuchElementException;

/** This is a DataSetIterator that is specialized for text corpora in XML formal.
 * It takes test and train dataset from SR XML files
 * Inputs/features: 
 * Labels/target: selected at class instantiation, Technology (from XML Tags) by default
 * @author Oleg Urzhumtcev
 */
public class XMLIterator implements DataSetIterator {
    /**
	 * Rev 1.0
	 */
	private static final long serialVersionUID = 1L;
	private final WordVectors wordVectors;
	private List<Pair<String, List<String>>> data;
	private ArrayList<String> labels = new ArrayList<String>();
    private final int batchSize = 50;
    private final int vectorSize;
    private final int truncateLength = 300;

    private int cursor = 0;
    private final File[] positiveFiles;
    private final File[] negativeFiles;
    private final TokenizerFactory tokenizerFactory;

    /**
     * @param dataDirectory the directory of the IMDB review data set
     * @param wordVectors WordVectors object
     * @param batchSize Size of each minibatch for training
     * @param truncateLength If reviews exceed
     * @param train If true: return the training data. If false: return the testing data.
     */
    public XMLIterator(List<Pair<String, List<String>>> ds, WordVectors wordVectors, boolean train) throws IOException {
        this.vectorSize = wordVectors.lookupTable().layerSize();
        this.wordVectors = wordVectors;
        this.data = ds;
        for (Pair<String, List<String>> d : data) if (!this.labels.contains(d.getLeft())) this.labels.add(d.getLeft());    
        
        String dataDirectory = "/Users/ourzhumt/MNIST/";
        File p = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/pos/") + "/");
        File n = new File(FilenameUtils.concat(dataDirectory, "aclImdb/" + (train ? "train" : "test") + "/neg/") + "/");
        positiveFiles = p.listFiles();
        negativeFiles = n.listFiles();

        tokenizerFactory = new DefaultTokenizerFactory();
        tokenizerFactory.setTokenPreProcessor(new CommonPreprocessor());
    }


    @Override
    public DataSet next(int num) {
        if (cursor >= data.size()) throw new NoSuchElementException();
        try{
            return nextCiscoDS(num);
        }catch(IOException e){
            throw new RuntimeException(e);
        }
    }
    
    private DataSet nextCiscoDS(int num) throws IOException {
        //Second: tokenize reviews and filter out unknown words
        //List<List<String>> allTokens = new ArrayList<>(num);
        int maxLength = 0;
        List<String> reviews = new ArrayList<>(num);
        List<Integer> rLabels = new ArrayList<>(num);
        for( int i=0; i<num && cursor<totalExamples(); i++ ){
            reviews.add(data.get(cursor).getRight().get(0));
            rLabels.add(this.labels.indexOf(data.get(cursor).getLeft()));
            cursor++;
        }

        List<List<String>> allTokens = new ArrayList<>(reviews.size());
        for(String s : reviews){
            List<String> tokens = tokenizerFactory.create(s).getTokens();
            List<String> tokensFiltered = new ArrayList<>();
            for(String t : tokens ){
                if(wordVectors.hasWord(t)) tokensFiltered.add(t);
            }
            allTokens.add(tokensFiltered);
            maxLength = Math.max(maxLength,tokensFiltered.size());
        }
        //If longest review exceeds 'truncateLength': only take the first 'truncateLength' words
        if(maxLength > truncateLength) maxLength = truncateLength;

        INDArray features = Nd4j.create(num, vectorSize);
        INDArray labels = Nd4j.create(num, this.labels.size(), maxLength);    //labels of N types
        //Because we are dealing with reviews of different lengths and only one output at the final time step: use padding arrays
        //Mask arrays contain 1 if data is present at that time step for that example, or 0 if data is just padding
        INDArray featuresMask = Nd4j.zeros(num, maxLength);
        INDArray labelsMask = Nd4j.zeros(num, maxLength);

        int[] temp = new int[2];
        int iBegin = cursor;
        for(int i=0; i < Math.min(num, data.size() - iBegin); i++ ){
            List<String> tokens = allTokens.get(i);
        	int idx = rLabels.get(i);	//+ Numeric class
        	
            temp[0] = i;
            //Get word vectors for each word in review, and put them in the training data
            for( int j=0; j<tokens.size() && j<maxLength && j < 1; j++ ){
                String token = tokens.get(j);
                INDArray vector = wordVectors.getWordVectorMatrix(token);
                features.put(new INDArrayIndex[]{NDArrayIndex.point(i), NDArrayIndex.all()}, vector);

                temp[1] = j;
                featuresMask.putScalar(temp, 1.0);  //Word is present (not padding) for this example + time step -> 1.0 in features mask
            }

            int lastIdx = Math.min(tokens.size(),maxLength);
            labels.putScalar(new int[]{i,idx,lastIdx-1},1.0);   //Set label: [0,1] for negative, [1,0] for positive
            labelsMask.putScalar(new int[]{i,lastIdx-1},1.0);   //Specify that an output exists at the final time step for this example
        }

        return new DataSet(features,labels,featuresMask,labelsMask);
    }	// nextCiscoDS

    @Override
    public int totalExamples() {
        return data.size();
    }

    @Override
    public int inputColumns() {
        return vectorSize;
    }

    @Override
    public int totalOutcomes() {
        return 2;
    }

    @Override
    public void reset() {
        cursor = 0;
    }

    @Override
    public int batch() {
        return batchSize;
    }

    @Override
    public int cursor() {
        return cursor;
    }

    @Override
    public int numExamples() {
        return totalExamples();
    }

    @Override
    public void setPreProcessor(DataSetPreProcessor preProcessor) {
        throw new UnsupportedOperationException();
    }

    @Override
    public List<String> getLabels() {
        return Arrays.asList("positive","negative");
    }

    @Override
    public boolean hasNext() {
        return cursor < numExamples();
    }

    @Override
    public DataSet next() {
        return next(batchSize);
    }

    @Override
    public void remove() {

    }

    /** Convenience method for loading review to String */
    public String loadReviewToString(int index) throws IOException{
        File f;
        if(index%2 == 0) f = positiveFiles[index/2];
        else f = negativeFiles[index/2];
        return FileUtils.readFileToString(f);
    }

    /** Convenience method to get label for review */
    public boolean isPositiveReview(int index){
        return index%2 == 0;
    }
}	// class