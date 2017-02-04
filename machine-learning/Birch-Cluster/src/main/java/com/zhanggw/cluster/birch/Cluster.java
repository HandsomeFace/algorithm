package com.zhanggw.cluster.birch;

import java.util.ArrayList;

/**
 * Ҷ�ӽڵ��е�С��Ⱥ
 * @author lyq
 *
 */
public class Cluster extends ClusteringFeature{
	//��Ⱥ�е����ݵ�
	private ArrayList<double[]> data;
	//���׽ڵ�
	private LeafNode parentNode;
	
	public Cluster(String[] record){
		data = new ArrayList<double[]>();
		
		double[] d = new double[record.length];
		for(int i=0; i<record.length; i++){
			d[i] = Double.parseDouble(record[i]);
		}
		data.add(d);
		//����CF��������
		setFeatures(data);
	}

	@Override
	protected void directAddCluster(ClusteringFeature node) {
		//����Ǿ���������ݼ�¼������ϲ����ݼ�¼
		Cluster c = (Cluster)node;
		ArrayList<double[]> dataRecords = c.getData();
		this.data.addAll(dataRecords);
		
		super.directAddCluster(node);
	}
	
	@Override
	public void addingCluster(ClusteringFeature clusteringFeature) {
		// TODO Auto-generated method stub
		
	}

	public LeafNode getParentNode() {
		return parentNode;
	}

	public void setParentNode(LeafNode parentNode) {
		this.parentNode = parentNode;
	}

	public ArrayList<double[]> getData() {
		return data;
	}

	public void setData(ArrayList<double[]> data) {
		this.data = data;
	}
}
