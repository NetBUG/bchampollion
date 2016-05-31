package org.netbug.bivec_gloss
import org.canova.api.io.WritableConverter
import org.canova.api.split.StringSplit
import org.canova.api.records.reader.RecordReader
//import org.canova.api.records.reader.RecordReaderAda
//import org.deeplearning4j.spark.sql.sources.canova
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator

class INDReader {
  def readDataSet(v1: String) = {
    //val rr = new RecordReader(new StringSplit(v1))
    /*val rr = new RecordReader(new StringSplit(v1))
    val dataSets: List[DataSet] = List()
    for(int j = 0; j < currList.size(); j++) {
        if(labelIndex >= 0 && j == labelIndex) {
            if(numPossibleLabels < 1)
                throw new IllegalStateException("Number of possible labels invalid, must be >= 1");
            Writable current = currList.get(j);
            if(converter != null)
                current = converter.convert(current);
            label = FeatureUtil.toOutcomeVector(Double.valueOf(current.toString()).intValue(), numPossibleLabels);
        }
        else {
            Writable current = currList.get(j);
            featureVector.putScalar(count++,Double.valueOf(current.toString()));
        }
    }

    dataSets.add(new DataSet(featureVector,labelIndex >= 0 ? label : featureVector));

    List<INDArray> inputs = new ArrayList<>();
    List<INDArray> labels = new ArrayList<>();
    for(DataSet data : dataSets) {
        inputs.add(data.getFeatureMatrix());
        labels.add(data.getLabels());
    }
	*/
	  val inputs:Array[Float] = Array()
	  val labels:Array[Float] = Array()
    val ret = new DataSet(Nd4j.vstack( Nd4j.create(inputs)),Nd4j.vstack( Nd4j.create(labels)));
    ret  
  }
  
  
}  // INDReader