package org.netbug.bivec_gloss

import com.github.fommil.netlib.BLAS.{ getInstance => blas }
import java.io.StringReader
import java.io._
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import org.apache.spark.{ SparkContext, SparkConf }
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.linalg.Vector
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors
import org.deeplearning4j.scaleout.aggregator.INDArrayAggregator
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.factory.Nd4j
import org.nd4j.linalg.indexing.INDArrayIndex;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator
import org.nd4j.linalg.lossfunctions.LossFunctions
import org.nd4j.linalg.lossfunctions.LossFunctions._
import org.slf4j.LoggerFactory
import scala.collection.JavaConverters._
import scala.collection.JavaConversions._
import scala.collection.mutable.{ ArrayBuffer, HashMap }
import scala.xml.{ XML, Elem }

/** @author Urzhumtcev Oleg
 *  @filename BivecGloss.scala
 *  @brief Class to build glossary from 
*/

class BivecDict(
  val model: Map[String, Array[Float]]) extends Serializable {

  private def cosineSimilarity(v1: Array[Float], v2: Array[Float]): Double = {
    require(v1.length == v2.length, "Vectors should have the same length")
    val n = v1.length
    val norm1 = blas.snrm2(n, v1, 1)
    val norm2 = blas.snrm2(n, v2, 1)
    if (norm1 == 0 || norm2 == 0) return 0.0
    blas.sdot(n, v1, 1, v2, 1) / norm1 / norm2
  }

  def distance(word1: String, word2: String) = {
    val v1 = transform(word1).toArray.map(_.toFloat)
    val v2 = transform(word2).toArray.map(_.toFloat)

    cosineSimilarity(v1, v2)

  }

  /**
   * Transforms a word to its vector representation
   * @param word a word
   * @return vector representation of word
   */
  def transform(word: String): Vector = {
    model.get(word) match {
      case Some(vec) =>
        Vectors.dense(vec.map(_.toDouble))
      case None =>
        throw new IllegalStateException(s"$word not in vocabulary")
    }
  }

  def exist(word: String): Boolean = {
    model.get(word) match {
      case Some(vec) => true
      case None      => false
    }
  }

  /**
   * Find synonyms of a word
   * @param word a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  def findSynonyms(word: String, num: Int): Array[(String, Double)] = {
    val vector = transform(word)
    findSynonyms(vector, num)
  }

  /**
   * Find synonyms of a word
   * @param word a word
   * @param threshold only consider words with cosine similarity above this threshold
   * @return array of (word, cosineSimilarity)
   */
  def findSynonymsAboveThreshold(word: String, threshold: Double = 0.5): Array[(String, Double)] = {
    val vector = transform(word)
    val fVector = vector.toArray.map(_.toFloat)
    model.mapValues(vec => cosineSimilarity(fVector, vec))
      .filter(x => x._2 >= threshold)
      .toArray
  }

  /**
   * Find synonyms of the vector representation of a word
   * @param vector vector representation of a word
   * @param num number of synonyms to find
   * @return array of (word, cosineSimilarity)
   */
  def findSynonyms(vector: Vector, num: Int): Array[(String, Double)] = {
    require(num > 0, "Number of similar words should > 0")
    // TODO: optimize top-k
    val fVector = vector.toArray.map(_.toFloat)
    model.mapValues(vec => cosineSimilarity(fVector, vec))
      .toSeq
      .sortBy(-_._2)
      .take(num + 1)
      .tail
      .toArray
  }

  /**
   * Returns a map of words to their vector representations.
   */
  def getVectors: Map[String, Array[Float]] = {
    model
  }
}

class BivecGloss (model_path: String, sc: SparkContext) {
  val SYNONYM_COUNT = 10
  val dictTxt = sc.textFile(model_path)
  //~ yields ("en", "de")
  val langs = sc.files.filter { x => x.contains("out.") && x.length < 7 }
  val dictMap = dictTxt.map { x => x.split(""" \|#\| """) }.filter { x => isValid(x) }.filter { x => goodToken(x(0), stopWords.value) }
      .map { x => x(0) -> x(1).split(",").map { x => x.toFloat } }
  val vecModel = new BivecDict(dictMap.collect().toMap)
  val vecModelB = sc.broadcast(vecModel)
  
  val dicSrc = sc.textFile(model_path.replace("xx", langs(0)))
  val dicTrg = sc.textFile(model_path.replace("xx", langs(1)))
  /**
  * @brief Gets k translations from Bivec model for each word
  */
  def knn_trans() = {
    vecModel.getVectors().map(x => (x._1, vecModelB.findSynonyms(x._1))) 
  }
  
  /**
  *  @brief Determines the threshold
  */
  def thresh_detect() = {
    val dic = vecModel.getVectors().map(x => x._1)
    val thresholds = dic.map(token => {
      val wordSyn = vecModel.findSynonyms(token, SYNONYM_COUNT)
      val threshold = wordSyn.map(synonym => {if (dicSrc.contains(synonym._1)) synonym._2 else -1 })
        .filter(x => x > 0)
      threshold(0)     //~ Take the first one
    })
  }
  
  /**
  *  @brief Gets k translations from Bivec model for each word
  */
  def prune_gloss(t: Double = 0.7) = {
    vecModel.getVectors().map(x => (x._1, vecModelB.findSynonymsAboveThreshold(x._1, t)))
  }
  
}

object BivecGlossTest {
  //~ One can use monomingual models as well
  val MODEL_PATH = "../bivec/out_mono_300/vec/out.sumvec.xx"
  
  def main(args: Array[String]): Unit = {
    val sc = new SparkContext(new SparkConf().setAppName("SRSimilarityTagsPR")
        .set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        .set("spark.kryoserializer.buffer", "512")
        .registerKryoClasses(Array(classOf[String]))
        .set("spark.app.id", this.getClass.getName)
        .set("spark.driver.maxResultSize", "4g"))
    val bg = BivecGloss(MODEL_PATH, sc)
    val thresh = thresh_detect()
    println("Threshold for " + bg.langs(0) + " and " + bg.langs(1) + ": " + thresh.toString())
  }  
}