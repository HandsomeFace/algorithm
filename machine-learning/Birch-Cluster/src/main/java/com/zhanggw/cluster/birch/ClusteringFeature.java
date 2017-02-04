package com.zhanggw.cluster.birch;

import java.util.ArrayList;

/**
 * ����������������
 * 
 * @author lyq
 * 
 */
public abstract class ClusteringFeature {
	// �����нڵ������Ŀ
	protected int N;
	// ������N���ڵ�����Ժ�
	protected double[] LS;
	// ������N���ڵ��ƽ����
	protected double[] SS;
	//�ڵ���ȣ�����CF�������
	protected int level;

	/**
	 * ���ݽڵ����ݹ���ClusterFeature
	 * 
	 * @param dataRecords
	 *            �ڵ����ݼ�¼
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
	 * CF���������ĵ��ӣ����뿼�ǻ���
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

		// 3�������Ͻ��е���
		for (int i = 0; i < LS.length; i++) {
			LS[i] += otherLS[i];
			SS[i] += otherSS[i];
		}
		this.N += N;
	}

	/**
	 * ��������֮��ľ��뼴������֮��ľ���
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
	 * ������ڶ����ƽ������
	 * 
	 * @param records
	 *            ���ڵ����ݼ�¼
	 * @return
	 */
	public static double computerInClusterDistance(ArrayList<double[]> records) {
		double sumDistance = 0;
		double[] data1;
		double[] data2;
		// ��������
		int totalNum = records.size();

		for (int i = 0; i < totalNum - 1; i++) {
			data1 = records.get(i);
			for (int j = i + 1; j < totalNum; j++) {
				data2 = records.get(j);
				sumDistance += computeOuDistance(data1, data2);
			}
		}

		// ���ص�ֵ�����ܶ������ܶ���Ӧ���룬���ظ���һ��
		return Math.sqrt(sumDistance / (totalNum * (totalNum - 1) / 2));
	}

	/**
	 * �Ը�����2������������ŷʽ����
	 * 
	 * @param record1
	 *            ������1
	 * @param record2
	 *            ������2
	 */
	private static double computeOuDistance(double[] record1, double[] record2) {
		double distance = 0;

		for (int i = 0; i < record1.length; i++) {
			distance += (record1[i] - record2[i]) * (record1[i] - record2[i]);
		}

		return distance;
	}

	/**
	 * ������ӽڵ������������ֵ���з��ѵĲ���
	 * 
	 * @param clusteringFeature
	 *            ����Ӿ۴�
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
