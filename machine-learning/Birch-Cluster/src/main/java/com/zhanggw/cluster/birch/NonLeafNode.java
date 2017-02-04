package com.zhanggw.cluster.birch;

import java.util.ArrayList;
import java.util.LinkedList;

/**
 * 非叶子节点
 * 
 * @author lyq
 * 
 */
public class NonLeafNode extends ClusteringFeature {
	// 非叶子节点的孩子节点可能为非叶子节点，也可能为叶子节点，要么是叶子节点，要么是非叶子节点，不会是两者的混合
	private ArrayList<NonLeafNode> nonLeafChilds;
	// 如果是叶子节点的孩子，则以双向链表的形式存在
	private LinkedList<LeafNode> leafChilds;
	// 父亲节点
	private NonLeafNode parentNode;

	/**
	 * 添加叶子节点
	 * 
	 * @param leafNode
	 *            待添加叶子节点
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
			// 如果添加后，叶子节点数超过平衡因子，则添加后需要分裂
			if (leafChilds.size() > BIRCHTool.B) {
				needDivided = true;
			}
		}

		return needDivided;
	}

	/**
	 * 添加非叶子节点
	 * 
	 * @param nonLeafNode
	 *            待添加非叶子节点
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
			// 如果添加后，节点数超过平衡因子，则添加失败
			if (nonLeafChilds.size() > BIRCHTool.B) {
				needDivided = true;
			}
		}

		return needDivided;
	}

	/**
	 * 因为叶子节点数超过阈值，进行分裂
	 * 
	 * @return
	 */
	public NonLeafNode[] leafNodeDivided() {
		NonLeafNode[] nonLeafNodes = new NonLeafNode[2];

		// 簇间距离差距最大的2个簇，后面的簇按照就近原则划分即可
		LeafNode node1 = null;
		LeafNode node2 = null;
		LeafNode tempNode = null;
		double maxValue = 0;
		double temp = 0;

		// 找出簇心距离差距最大的2个簇
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
		// 就近分配簇
		for (LeafNode c : leafChilds) {
			if (ClusteringFeature.computerClusterDistance(c, node1) < ClusteringFeature.computerClusterDistance(c, node2)) {
				// 簇间距离如果接近最小簇，就加入最小簇所属叶子节点
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
	 * 因为非叶子节点数超过阈值，进行分裂
	 * 
	 * @return
	 */
	public NonLeafNode[] nonLeafNodeDivided() {
		NonLeafNode[] nonLeafNodes = new NonLeafNode[2];

		// 簇间距离差距最大的2个簇，后面的簇按照就近原则划分即可
		NonLeafNode node1 = null;
		NonLeafNode node2 = null;
		NonLeafNode tempNode = null;
		double maxValue = 0;
		double temp = 0;

		// 找出簇心距离差距最大的2个簇
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
		// 就近分配簇
		for (NonLeafNode c : nonLeafChilds) {
			if (ClusteringFeature.computerClusterDistance(c, node1) < ClusteringFeature.computerClusterDistance(c, node2)) {
				// 簇间距离如果接近最小簇，就加入最小簇所属叶子节点
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
	 * 寻找到最接近的叶子节点
	 * 
	 * @param cluster
	 *            待添加聚簇
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

			// 递归继续往下找
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
		// 更新聚类特征值
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
