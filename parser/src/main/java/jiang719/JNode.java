package jiang719;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Queue;
import java.util.Stack;
import java.util.concurrent.LinkedBlockingQueue;

import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;

public class JNode implements Comparable<JNode> {
    public Node node;
	public final int depth;
	public JNode father;
	public ArrayList<JNode> children = new ArrayList<JNode>();
	
	public JNode(Node node, int depth){
		this.node = node;
		this.depth = depth;
	}
	
	public void addChild(JNode child) {
		this.children.add(child);
		child.addFather(this);
	}
	
	public void addFather(JNode father) {
		this.father = father;
	}
	
	public Range getRange() {
		return this.node.getRange().get();
	}
	
	public ArrayList<JNode> getChildrenByTpye(Class<?> clazz) {
		ArrayList<JNode> children = new ArrayList<JNode>();
		for (JNode child : this.children) {
			if (child.node.getClass().equals(clazz)) {
				children.add(child);
			}
		}
		return children;
	}
	
	public boolean classEquals(Class<?> clazz) {
		return this.node.getClass().equals(clazz);
	}
	
	public String getValue() {
		return this.node.toString();
	}
	
	public String getNodeClass() {
		String classname = this.node.getClass().toString();
		return classname.substring(classname.lastIndexOf(".") + 1);
	}
	
	public void sortChildren(JNode node) {
		Collections.sort(node.children);
		for (JNode child : node.children)
			sortChildren(child);
	}
	
	public ArrayList<JNode> BFS() {
		sortChildren(this);
		
		ArrayList<JNode> bfs = new ArrayList<JNode>();
		Queue<JNode> queue = new LinkedBlockingQueue<JNode>();
		queue.add(this);
		while (! queue.isEmpty()) {
			JNode cur = queue.poll();
			for (JNode child : cur.children) {
				queue.add(child);
			}
			
			bfs.add(cur);
		}
		return bfs;
	}
	
	public ArrayList<JNode> DFS(){
		sortChildren(this);
		
		ArrayList<JNode> dfs = new ArrayList<JNode>();
		Stack<JNode> stack = new Stack<JNode>();
		stack.push(this);
		while (! stack.isEmpty()) {
			JNode cur = stack.pop();
			for (int i = cur.children.size() - 1; i >= 0; i -= 1) {
				stack.push(cur.children.get(i));
			}
			
			dfs.add(cur);
		}
		return dfs;
	}
	
	public void print() {
		for (JNode JNode : this.DFS()) {
			System.out.println(JNode);
		}
	}
	
	public String toString() {
		return this.depth + " " + this.getNodeClass() + " " + this.children.size() + "\n" + 
				this.node.getRange().get().toString() + "\n" + this.node.toString();
	}

	public int compareTo(JNode JNode) {
		if (! this.node.getRange().isPresent())
			return 1;
		if (! JNode.node.getRange().isPresent())
			return -1;
		Range r1 = this.node.getRange().get();
		Range r2 = JNode.node.getRange().get();
		if (r1.begin.isBefore(r2.begin))
			return -1;
		else if (r1.begin.equals(r2.begin) && r1.end.isBefore(r2.end))
			return -1;
		else
			return 1;
	}
}
