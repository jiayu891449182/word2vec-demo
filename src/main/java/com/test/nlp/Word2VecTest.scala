package com.test.nlp

import org.apache.spark.mllib.feature.Word2Vec
import org.apache.spark.{SparkConf, SparkContext}

object Word2VecTest {

  def main(args : Array[String]) : Unit = {
    val sc : SparkContext = new SparkContext(new SparkConf)
    val input = sc.textFile(args(0))
      .map(line => line.split(" ").toSeq)
    val word2vec = new Word2Vec()
    word2vec.setSeed(100)
    val word2VecModel = word2vec.fit(input)
    val synonyms = word2VecModel.findSynonyms("北京", 5)
    println(synonyms.mkString(","))
  }
}
