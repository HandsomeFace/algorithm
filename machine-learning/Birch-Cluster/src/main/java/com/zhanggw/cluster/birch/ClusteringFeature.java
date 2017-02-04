package com.zhanggw.cluster.birch;

import java.util.ArrayList;

/**
 * 聚类特征基本属性
 * 
 * @author lyq
 * 
 */
public abstract class ClusteringFeature {
	// 子类中节点的总数目
	protected int N;
	// 子类中N个节点的线性和
	protected double[] LS;
	// 子类中N个节点的平方和
	protected double[] SS;
	//节点深度，用于CF树的输出
	protected int level;

	/**
	 * 根据节点数据构造ClusterFeature
	 * 
	 * @param dataRecords
	 *            节点数据记录
	 */
	protected void setFeatures(ArrayList<double[]> dataRecords) {
		this.N = dataRecords.size();
		int num = dataRecords.get(0).length;
		double[] record;
		LS = new double[num];
		SS = new double[num];
		for (int j = 0; j < num; j++) {
			LS[j] = 0;
			SS[j] = 0;
		}

		for (int i = 0; i < dataRecords.size(); i++) {
			record = dataRecords.get(i);
			for (int j = 0; j < record.length; j++) {
				LS[j] += record[j];
				SS[j] += record[j] * record[j];
			}
		}
	}

	/**
	 * CF向量特征的叠加，无须考虑划分
	 * 
	 * @param node
	 */
	protected void directAddCluster(ClusteringFeature node) {
		int N = node.getN();
		double[] otherLS = node.getLS();
		double[] otherSS = node.getSS();
		
		if(LS == null){
			this.N = 0;
			LS = new double[otherLS.length];
			SS = new double[otherLS.length];
			
			for(int i=0; i<LS.length; i++){
				LS[i] = 0;
				SS[i] = 0;
			}
		}

		// 3个数量上进行叠加
		for (int i = 0; i < LS.length; i++) {
			LS[i] += otherLS[i];
			SS[i] += otherSS[i];
		}
		this.N += N;
	}

	/**
	 * 计算簇与簇之间的距离即簇中心之间的距离
	 * 
	 * @return
	 */
	public static double computerClusterDistance(ClusteringFeature first, ClusteringFeature other) {
		double distance = 0;
		
		int firstNum = first.N;
		int otherNum = other.N;

		for (int i = 0; i < first.LS.length; i++) {
			distance += (first.LS[i] / firstNum - other.LS[i] / otherNum) * (first.LS[i] / firstNum - other.LS[i] / otherNum);
		}
		distance = Math.sqrt(distance);

		return distance;
	}

	/**
	 * 计算簇内对象的平均距离
	 * 
	 * @param records
	 *            簇内的数据记录
	 * @return
	 */
	public static double computerInClusterDistance(ArrayList<double[]> records) {
		double sumDistance = 0;
		double[] data1;
		double[] data2;
		// 数据总数
		int totalNum = records.size();

		for (int i = 0; i < totalNum - 1; i++) {
			data1 = records.get(i);
			for (int j = i + 1; j < totalNum; j++) {
				data2 = records.get(j);
				sumDistance += computeOuDistance(data1, data2);
			}
		}

		// 返回的值除以总对数，总对数应减半，会重复算一次
		return Math.sqrt(sumDistance / (totalNum * (totalNum - 1) / 2));
	}

	/**
	 * 对给定的2个向量，计算欧式距离
	 * 
	 * @param record1
	 *            向量点1
	 * @param record2
	 *            向量点2
	 */
	private static double computeOuDistance(double[] record1, double[] record2) {
		double distance = 0;

		for (int i = 0; i < record1.length; i++) {
			distance += (record1[i] - record2[i]) * (record1[i] - record2[i]);
		}

		return distance;
	}

	/**
	 * 聚类添加节点包括，超出阈值进行分裂的操作
	 * 
	 * @param clusteringFeature
	 *            待添加聚簇
	 */
	public abstract void addingCluster(ClusteringFeature clusteringFeature);
	
	public int getN() {
		return N;
	}

	public void setN(int n) {
		N = n;
	}

	public double[] getLS() {
		return LS;
	}

	public void setLS(double[] lS) {
		LS = lS;
	}

	public double[] getSS() {
		return SS;
	}

	public void setSS(double[] sS) {
		SS = sS;
	}
	
	public int getLevel() {
		return level;
	}

	public void setLevel(int level) {
		this.level = level;
	}
}
