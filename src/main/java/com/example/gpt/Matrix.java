package com.example.gpt;

public class Matrix {
    private final double[][] data;
    private final int rows;
    private final int cols;

    public Matrix(int rows, int cols) {
        this.rows = rows;
        this.cols = cols;
        this.data = new double[rows][cols];
    }

    public Matrix(double[][] data) {
        this.rows = data.length;
        this.cols = data[0].length;
        this.data = data;
    }

    public int getRows() {
        return rows;
    }

    public int getCols() {
        return cols;
    }

    public double[][] getData() {
        return data;
    }

    public static Matrix random(int rows, int cols, double mean, double std) {
        Matrix result = new Matrix(rows, cols);
        java.util.Random rand = new java.util.Random();
        for (int i = 0; i < rows; i++) {
            double[] rowData = result.data[i];
            for (int j = 0; j < cols; j++) {
                rowData[j] = mean + std * rand.nextGaussian();
            }
        }
        return result;
    }

    // Subtracts another matrix with broadcasting support
    public Matrix subtract(Matrix other) {
        if (this.rows == other.rows && this.cols == other.cols) {
            // Element-wise subtraction
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                double[] thisRow = this.data[i];
                double[] otherRow = other.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] - otherRow[j];
                }
            }
            return result;
        } else if (other.rows == this.rows && other.cols == 1) {
            // Subtracting column vector from each row
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                double value = other.data[i][0];
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] - value;
                }
            }
            return result;
        } else if (other.rows == 1 && other.cols == this.cols) {
            // Subtracting row vector from each row
            Matrix result = new Matrix(this.rows, this.cols);
            double[] rowValues = other.data[0];
            for (int i = 0; i < this.rows; i++) {
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] - rowValues[j];
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("Matrix dimensions are incompatible for subtraction.");
        }
    }

    public Matrix subtract(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < cols; j++) {
                resultRow[j] = thisRow[j] - scalar;
            }
        }
        return result;
    }

    // Adds another matrix with broadcasting support
    public Matrix add(Matrix other) {
        if (this.rows == other.rows && this.cols == other.cols) {
            // Element-wise addition
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                double[] thisRow = this.data[i];
                double[] otherRow = other.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] + otherRow[j];
                }
            }
            return result;
        } else if (other.rows == this.rows && other.cols == 1) {
            // Adding column vector to each row
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                double value = other.data[i][0];
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] + value;
                }
            }
            return result;
        } else if (other.rows == 1 && other.cols == this.cols) {
            // Adding row vector to each row
            Matrix result = new Matrix(this.rows, this.cols);
            double[] rowValues = other.data[0];
            for (int i = 0; i < this.rows; i++) {
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] + rowValues[j];
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("Matrix dimensions are incompatible for addition.");
        }
    }

    public Matrix add(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < cols; j++) {
                resultRow[j] = thisRow[j] + scalar;
            }
        }
        return result;
    }

    // Element-wise square root
    public static Matrix sqrt(Matrix m) {
        Matrix result = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < m.cols; j++) {
                resultRow[j] = Math.sqrt(mRow[j]);
            }
        }
        return result;
    }

    // Mean along a dimension (-1 for row-wise mean)
    public Matrix mean(int axis) {
        if (axis == -1) { // Row-wise mean
            Matrix result = new Matrix(this.rows, 1);
            for (int i = 0; i < rows; i++) {
                double sum = 0.0;
                double[] thisRow = this.data[i];
                for (int j = 0; j < cols; j++) {
                    sum += thisRow[j];
                }
                result.data[i][0] = sum / cols;
            }
            return result;
        }
        throw new UnsupportedOperationException("Mean only supports axis=-1 (row-wise) for now.");
    }

    // Variance along a dimension (-1 for row-wise variance)
    public Matrix variance(int axis, boolean unbiased) {
        if (axis == -1) { // Row-wise variance
            Matrix mean = this.mean(-1);
            Matrix result = new Matrix(this.rows, 1);
            for (int i = 0; i < rows; i++) {
                double sum = 0.0;
                double meanValue = mean.data[i][0];
                double[] thisRow = this.data[i];
                for (int j = 0; j < cols; j++) {
                    double diff = thisRow[j] - meanValue;
                    sum += diff * diff;
                }
                result.data[i][0] = sum / (unbiased ? (cols - 1) : cols);
            }
            return result;
        }
        throw new UnsupportedOperationException("Variance only supports axis=-1 (row-wise) for now.");
    }

    // Divides by another matrix with broadcasting support
    public Matrix divide(Matrix other) {
        if (this.rows == other.rows && this.cols == other.cols) {
            // Element-wise division
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                double[] thisRow = this.data[i];
                double[] otherRow = other.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] / otherRow[j];
                }
            }
            return result;
        } else if (other.rows == this.rows && other.cols == 1) {
            // Dividing by column vector
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                double value = other.data[i][0];
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] / value;
                }
            }
            return result;
        } else if (other.rows == 1 && other.cols == this.cols) {
            // Dividing by row vector
            Matrix result = new Matrix(this.rows, this.cols);
            double[] rowValues = other.data[0];
            for (int i = 0; i < this.rows; i++) {
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] / rowValues[j];
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("Matrix dimensions are incompatible for division.");
        }
    }

    public Matrix divide(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        double invScalar = 1.0 / scalar;
        for (int i = 0; i < rows; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < cols; j++) {
                resultRow[j] = thisRow[j] * invScalar;
            }
        }
        return result;
    }

    // Element-wise multiplication with broadcasting support
    public Matrix multiply(Matrix other) {
        if (this.rows == other.rows && this.cols == other.cols) {
            // Element-wise multiplication
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < rows; i++) {
                double[] thisRow = this.data[i];
                double[] otherRow = other.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < cols; j++) {
                    resultRow[j] = thisRow[j] * otherRow[j];
                }
            }
            return result;
        } else if (other.rows == this.rows && other.cols == 1) {
            // Multiply by column vector
            Matrix result = new Matrix(this.rows, this.cols);
            for (int i = 0; i < this.rows; i++) {
                double value = other.data[i][0];
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] * value;
                }
            }
            return result;
        } else if (other.rows == 1 && other.cols == this.cols) {
            // Multiply by row vector
            Matrix result = new Matrix(this.rows, this.cols);
            double[] rowValues = other.data[0];
            for (int i = 0; i < this.rows; i++) {
                double[] thisRow = this.data[i];
                double[] resultRow = result.data[i];
                for (int j = 0; j < this.cols; j++) {
                    resultRow[j] = thisRow[j] * rowValues[j];
                }
            }
            return result;
        } else {
            throw new IllegalArgumentException("Matrix dimensions are incompatible for multiplication.");
        }
    }

    public Matrix multiply(double scalar) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < rows; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < cols; j++) {
                resultRow[j] = thisRow[j] * scalar;
            }
        }
        return result;
    }

    public Matrix matMul(Matrix other) {
        if (this.cols != other.rows) {
            throw new IllegalArgumentException("Matrix dimensions are not compatible for multiplication.");
        }
        Matrix result = new Matrix(this.rows, other.cols);
        double[][] otherData = other.data;
        for (int i = 0; i < this.rows; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int k = 0; k < this.cols; k++) {
                double elemA = thisRow[k];
                double[] otherRow = otherData[k];
                for (int j = 0; j < other.cols; j++) {
                    resultRow[j] += elemA * otherRow[j];
                }
            }
        }
        return result;
    }

    public Matrix addRowVector(Matrix rowVector) {
        if (rowVector.rows != 1 || rowVector.cols != this.cols) {
            throw new IllegalArgumentException("Row vector dimensions must match matrix columns.");
        }
        return this.add(rowVector); 
    }

    public Matrix pow(double exponent) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < this.cols; j++) {
                resultRow[j] = Math.pow(thisRow[j], exponent);
            }
        }
        return result;
    }

    public Matrix applyFunction(java.util.function.Function<Double, Double> func) {
        Matrix result = new Matrix(this.rows, this.cols);
        for (int i = 0; i < this.rows; i++) {
            double[] thisRow = this.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < this.cols; j++) {
                resultRow[j] = func.apply(thisRow[j]);
            }
        }
        return result;
    }

    public Matrix reshape(int newRows, int newCols) {
        if (newRows * newCols != this.rows * this.cols) {
            throw new IllegalArgumentException("Total elements must remain the same during reshape.");
        }
        Matrix result = new Matrix(newRows, newCols);
        int index = 0;
        for (int i = 0; i < this.rows; i++) {
            double[] thisRow = this.data[i];
            for (int j = 0; j < this.cols; j++) {
                int newRow = index / newCols;
                int newCol = index % newCols;
                result.data[newRow][newCol] = thisRow[j];
                index++;
            }
        }
        return result;
    }

    public static Matrix transpose(Matrix m) {
        Matrix result = new Matrix(m.cols, m.rows);
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            for (int j = 0; j < m.cols; j++) {
                result.data[j][i] = mRow[j];
            }
        }
        return result;
    }

    public static Matrix applyMask(Matrix m, Matrix mask) {
        if (m.rows != mask.rows || m.cols != mask.cols) {
            throw new IllegalArgumentException("Mask dimensions must match matrix dimensions.");
        }
        Matrix result = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            double[] maskRow = mask.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < m.cols; j++) {
                resultRow[j] = maskRow[j] == 0 ? mRow[j] : -1e9;
            }
        }
        return result;
    }

    // Softmax function
    public static Matrix softmax(Matrix m) {
        Matrix result = new Matrix(m.rows, m.cols);
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            double[] resultRow = result.data[i];
            double max = Double.NEGATIVE_INFINITY;
            for (double v : mRow) {
                max = Math.max(max, v);
            }
            double sum = 0.0;
            for (int j = 0; j < m.cols; j++) {
                resultRow[j] = Math.exp(mRow[j] - max);
                sum += resultRow[j];
            }
            for (int j = 0; j < m.cols; j++) {
                resultRow[j] /= sum;
            }
        }
        return result;
    }

    public double[] getRow(int rowIndex) {
        if (rowIndex < 0 || rowIndex >= this.rows) {
            throw new IllegalArgumentException("Row index out of bounds.");
        }
        return data[rowIndex];
    }

    public Matrix getSubMatrix(int rowStart, int rowEnd, int colStart, int colEnd) {
        int newRows = rowEnd - rowStart;
        int newCols = colEnd - colStart;
        if (rowStart < 0 || rowEnd > this.rows || colStart < 0 || colEnd > this.cols) {
            throw new IllegalArgumentException("Invalid submatrix indices.");
        }
        Matrix result = new Matrix(newRows, newCols);
        for (int i = 0; i < newRows; i++) {
            System.arraycopy(this.data[rowStart + i], colStart, result.data[i], 0, newCols);
        }
        return result;
    }

    public void setData(double[][] newData) {
        if (newData.length != this.rows || newData[0].length != this.cols) {
            throw new IllegalArgumentException("Data dimensions do not match matrix dimensions.");
        }
        for (int i = 0; i < this.rows; i++) {
            System.arraycopy(newData[i], 0, this.data[i], 0, this.cols);
        }
    }

    // Applies random dropout with given rate
    public static Matrix dropout(Matrix m, double rate) {
        Matrix result = new Matrix(m.rows, m.cols);
        java.util.Random random = new java.util.Random();
        for (int i = 0; i < m.rows; i++) {
            double[] mRow = m.data[i];
            double[] resultRow = result.data[i];
            for (int j = 0; j < m.cols; j++) {
                resultRow[j] = random.nextDouble() > rate ? mRow[j] : 0.0;
            }
        }
        return result;
    }
}
