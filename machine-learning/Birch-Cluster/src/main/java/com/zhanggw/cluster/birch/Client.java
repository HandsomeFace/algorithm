package com.zhanggw.cluster.birch;

/**
 * BIRCH�����㷨������
 * @author lyq
 *
 */
public class Client {
	public static void main(String[] args){
		String filePath = "src/main/resource/testInput.txt";
		//�ڲ��ڵ�ƽ������B
		int B = 2;
		//Ҷ�ӽڵ�ƽ������L
		int L = 2;
		//��ֱ����ֵT
		double T = 0.6;
		
		BIRCHTool tool = new BIRCHTool(filePath, B, L, T);
		tool.startBuilding();
	}
}
