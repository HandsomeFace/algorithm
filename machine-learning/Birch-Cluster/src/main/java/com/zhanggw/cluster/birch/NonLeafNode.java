package com.zhanggw.cluster.birch;

import java.util.ArrayList;
import java.util.LinkedList;

/**
 * ��Ҷ�ӽڵ�
 * 
 * @author lyq
 * 
 */
public class NonLeafNode extends ClusteringFeature {
	// ��Ҷ�ӽڵ�ĺ��ӽڵ����Ϊ��Ҷ�ӽڵ㣬Ҳ����ΪҶ�ӽڵ㣬Ҫô��Ҷ�ӽڵ㣬Ҫô�Ƿ�Ҷ�ӽڵ㣬���������ߵĻ��
	private ArrayList<NonLeafNode> nonLeafChilds;
	// �����Ҷ�ӽڵ�ĺ��ӣ�����˫���������ʽ����
	private LinkedList<LeafNode> leafChilds;
	// ���׽ڵ�
	private NonLeafNode parentNode;

	/**
	 * ���Ҷ�ӽڵ�
	 * 
	 * @param leafNode
	 *            �����Ҷ�ӽڵ�
	 * @return
	 */
	public boolean addingNeededDivide(LeafNode leafNode) {
		boolean needDivided = false;
		if (leafChilds == null) {
			leafChilds = new LinkedList<>();
			leafChilds.add(leafNode);
			leafNode.setParentNode(this);
		} else {
			leafChilds.add(leafNode);
			leafNode.setParentNode(this);
			// �����Ӻ�Ҷ�ӽڵ�������ƽ�����ӣ�����Ӻ���Ҫ����
			if (leafChilds.size() > BIRCHTool.B) {
				needDivided = true;
			}
		}

		return needDivided;
	}

	/**
	 * ��ӷ�Ҷ�ӽڵ�
	 * 
	 * @param nonLeafNode
	 *            ����ӷ�Ҷ�ӽڵ�
	 * @return
	 */
	public boolean addingNeededDivide(NonLeafNode nonLeafNode) {
		boolean needDivided = false;
		if (nonLeafChilds == null) {
			nonLeafChilds = new ArrayList<>();
			nonLeafChilds.add(nonLeafNode);
			nonLeafNode.setParentNode(this);
		} else {
			nonLeafChilds.add(nonLeafNode);
			nonLeafNode.setParentNode(this);
			// �����Ӻ󣬽ڵ�������ƽ�����ӣ������ʧ��
			if (nonLeafChilds.size() > BIRCHTool.B) {
				needDivided = true;
			}
		}

		return needDivided;
	}

	/**
	 * ��ΪҶ�ӽڵ���������ֵ�����з���
	 * 
	 * @return
	 */
	public NonLeafNode[] leafNodeDivided() {
		NonLeafNode[] nonLeafNodes = new NonLeafNode[2];

		// �ؼ����������2���أ�����Ĵذ��վͽ�ԭ�򻮷ּ���
		LeafNode node1 = null;
		LeafNode node2 = null;
		LeafNode tempNode = null;
		double maxValue = 0;
		double temp = 0;

		// �ҳ����ľ���������2����
		for (int i = 0; i < leafChilds.size() - 1; i++) {
			tempNode = leafChilds.get(i);
			for (int j = i + 1; j < leafChilds.size(); j++) {
				temp = ClusteringFeature.computerClusterDistance(tempNode, leafChilds.get(j));

				if (temp > maxValue) {
					maxValue = temp;
					node1 = tempNode;
					node2 = leafChilds.get(j);
				}
			}
		}

		nonLeafNodes[0] = new NonLeafNode();
		nonLeafNodes[0].addingCluster(node1);
		nonLeafNodes[1] = new NonLeafNode();
		nonLeafNodes[1].addingCluster(node2);
		leafChilds.remove(node1);
		leafChilds.remove(node2);
		// �ͽ������
		for (LeafNode c : leafChilds) {
			if (ClusteringFeature.computerClusterDistance(c, node1) < ClusteringFeature.computerClusterDistance(c, node2)) {
				// �ؼ��������ӽ���С�أ��ͼ�����С������Ҷ�ӽڵ�
				nonLeafNodes[0].addingCluster(c);
				c.setParentNode(nonLeafNodes[0]);
			} else {
				nonLeafNodes[1].addingCluster(c);
				c.setParentNode(nonLeafNodes[1]);
			}
		}

		return nonLeafNodes;
	}

	/**
	 * ��Ϊ��Ҷ�ӽڵ���������ֵ�����з���
	 * 
	 * @return
	 */
	public NonLeafNode[] nonLeafNodeDivided() {
		NonLeafNode[] nonLeafNodes = new NonLeafNode[2];

		// �ؼ����������2���أ�����Ĵذ��վͽ�ԭ�򻮷ּ���
		NonLeafNode node1 = null;
		NonLeafNode node2 = null;
		NonLeafNode tempNode = null;
		double maxValue = 0;
		double temp = 0;

		// �ҳ����ľ���������2����
		for (int i = 0; i < nonLeafChilds.size() - 1; i++) {
			tempNode = nonLeafChilds.get(i);
			for (int j = i + 1; j < nonLeafChilds.size(); j++) {
				temp = ClusteringFeature.computerClusterDistance(tempNode, nonLeafChilds.get(j));

				if (temp > maxValue) {
					maxValue = temp;
					node1 = tempNode;
					node2 = nonLeafChilds.get(j);
				}
			}
		}

		nonLeafNodes[0] = new NonLeafNode();
		nonLeafNodes[0].addingCluster(node1);
		nonLeafNodes[1] = new NonLeafNode();
		nonLeafNodes[1].addingCluster(node2);
		nonLeafChilds.remove(node1);
		nonLeafChilds.remove(node2);
		// �ͽ������
		for (NonLeafNode c : nonLeafChilds) {
			if (ClusteringFeature.computerClusterDistance(c, node1) < ClusteringFeature.computerClusterDistance(c, node2)) {
				// �ؼ��������ӽ���С�أ��ͼ�����С������Ҷ�ӽڵ�
				nonLeafNodes[0].addingCluster(c);
				c.setParentNode(nonLeafNodes[0]);
			} else {
				nonLeafNodes[1].addingCluster(c);
				c.setParentNode(nonLeafNodes[1]);
			}
		}

		return nonLeafNodes;
	}

	/**
	 * Ѱ�ҵ���ӽ���Ҷ�ӽڵ�
	 * 
	 * @param cluster
	 *            ����Ӿ۴�
	 * @return
	 */
	public LeafNode findedClosestNode(Cluster cluster) {
		LeafNode node = null;
		NonLeafNode nonLeafNode = null;
		double temp;
		double distance = Integer.MAX_VALUE;

		if (nonLeafChilds == null) {
			for (LeafNode n : leafChilds) {
				temp = ClusteringFeature.computerClusterDistance(n, cluster);
				if (temp < distance) {
					distance = temp;
					node = n;
				}
			}
		} else {
			for (NonLeafNode n : nonLeafChilds) {
				temp = ClusteringFeature.computerClusterDistance(n, cluster);
				if (temp < distance) {
					distance = temp;
					nonLeafNode = n;
				}
			}

			// �ݹ����������
			node = nonLeafNode.findedClosestNode(cluster);
		}

		return node;
	}

	@Override
	public void addingCluster(ClusteringFeature clusteringFeature) {
		LeafNode leafNode = null;
		NonLeafNode nonLeafNode = null;
		NonLeafNode[] nonLeafNodeArrays;
		boolean neededDivide = false;
		// ���¾�������ֵ
		directAddCluster(clusteringFeature);

		if (clusteringFeature instanceof LeafNode) {
			leafNode = (LeafNode) clusteringFeature;
		} else {
			nonLeafNode = (NonLeafNode) clusteringFeature;
		}

		if (nonLeafNode != null) {
			neededDivide = addingNeededDivide(nonLeafNode);

			if (neededDivide) {
				if (parentNode == null) {
					parentNode = new NonLeafNode();
				} else {
					parentNode.nonLeafChilds.remove(this);
				}

				nonLeafNodeArrays = this.nonLeafNodeDivided();
				for (NonLeafNode n1 : nonLeafNodeArrays) {
					parentNode.addingCluster(n1);
				}
			}
		} else {
			neededDivide = addingNeededDivide(leafNode);

			if (neededDivide) {
				if (parentNode == null) {
					parentNode = new NonLeafNode();
				} else {
					parentNode.nonLeafChilds.remove(this);
				}

				nonLeafNodeArrays = this.leafNodeDivided();
				for (NonLeafNode n2 : nonLeafNodeArrays) {
					parentNode.addingCluster(n2);
				}
			}
		}
	}

	@Override
	protected void directAddCluster(ClusteringFeature node) {
		// TODO Auto-generated method stub
		if (parentNode != null) {
			parentNode.directAddCluster(node);
		}

		super.directAddCluster(node);
	}

	public ArrayList<NonLeafNode> getNonLeafChilds() {
		return nonLeafChilds;
	}

	public void setNonLeafChilds(ArrayList<NonLeafNode> nonLeafChilds) {
		this.nonLeafChilds = nonLeafChilds;
	}

	public LinkedList<LeafNode> getLeafChilds() {
		return leafChilds;
	}

	public void setLeafChilds(LinkedList<LeafNode> leafChilds) {
		this.leafChilds = leafChilds;
	}
	
	public NonLeafNode getParentNode() {
		return parentNode;
	}

	public void setParentNode(NonLeafNode parentNode) {
		this.parentNode = parentNode;
	}
	
}
