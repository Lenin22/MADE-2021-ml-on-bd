import breeze.linalg.DenseMatrix._
import breeze.linalg.norm._
import breeze.linalg._
import breeze.linalg.csvwrite

import java.io._

object LinearRegression extends App
{
  // parameters & vars
  var nIter: Int = 1000
  val theta = DenseVector.ones[Double](4)
  var alpha: Double = 0.005
  val debugger = new PrintWriter(new File("data/log.txt" ))

  def mse(X: DenseMatrix[Double], y: DenseVector[Double]): Double =
  {
    0.5 * norm(predict(X)-y) / y.size
  }

  def predict(X: DenseMatrix[Double]): DenseVector[Double] =
  {
    X * theta
  }

  def backProp(X: DenseMatrix[Double], y: DenseVector[Double]): Unit =
  {
    theta -= alpha / X.rows * (X.t *(predict(X)-y))
  }

  // adding ones for bias
  var xTrain: DenseMatrix[Double] = horzcat(csvread(new java.io.File("data/train.csv")),
    ones[Double](10000,1) )
  var xTest: DenseMatrix[Double] = horzcat(csvread(new java.io.File("data/test.csv")),
    ones[Double](1000,1) )

  val yTrain = csvread(new java.io.File("data/ytrain.csv"))

  var i:Int = 0
  while (i < nIter)
  {
    backProp(xTrain, yTrain.toDenseVector)
    val trainLoss: Double = mse(xTrain, yTrain.toDenseVector)

    debugger.println(s"Train loss: $trainLoss")

    i+=1
  }

  debugger.close
  csvwrite(new File("data/prediction.csv"), predict(xTest).toDenseMatrix)
}
