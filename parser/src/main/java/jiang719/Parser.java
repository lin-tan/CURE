package jiang719;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.*;

import com.github.javadocparser.TokenMgrError;
import com.github.javaparser.JavaParser;
import com.github.javaparser.Range;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.ImportDeclaration;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.PackageDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.Name;
import com.github.javaparser.ast.nodeTypes.*;


public class Parser {
	
	public static JavaParser parser = new JavaParser();
	
	public static CompilationUnit parseFile(String filename) throws IOException {
		try {
			Node root = parser.parse(new File(filename)).getResult().get();
			Parser.removeComments(root);
			return (CompilationUnit) root;
		} catch (Exception e) {
			return null;
		}
	}
	
	public static CompilationUnit parseFile(File file) throws FileNotFoundException {
		Node root = parser.parse(file).getResult().get();
		Parser.removeComments(root);
		return (CompilationUnit) root;
	}
	
	public static boolean containLines(Node node, int beginline, int endline, String subnodecode) {
		if (! node.getRange().isPresent())
			return false;
		Range noderange = node.getRange().get();
		if (noderange.begin.line > beginline || noderange.end.line < endline)
			return false;
		String nodestring = node.toString().replaceAll("\\s+", "");
		String subnodestring = subnodecode.replaceAll("\\s+", "");
		return nodestring.contains(subnodestring);
	}
	
	public static boolean containLines(Node node, String subnodecode) {
		if (! node.getRange().isPresent())
			return false;
		String nodestring = node.toString().replaceAll("\\s+", "");
		String subnodestring = subnodecode.replaceAll("\\s+", "");
		return nodestring.contains(subnodestring);
	}
	
	public static boolean containedByLines(Node node, int beginline, int endline, String code) {
		if (! node.getRange().isPresent())
			return false;
		Range noderange = node.getRange().get();
		if (noderange.begin.line < beginline || noderange.end.line > endline)
			return false;
		String nodestring = node.toString().replaceAll("\\s+", "");
		String string = code.replaceAll("\\s+", "");
		return string.contains(nodestring);
	}
	
	private static void findSubNode_(Node node, int beginline, int endline, String subnodecode, Node[] subnode) {
		if (containLines(node, beginline, endline, subnodecode)) {
			subnode[0] = node;
			List<Node> childs = node.getChildNodes();
			for (Node child : childs) {
				findSubNode_(child, beginline, endline, subnodecode, subnode);
			}
		}
	}
	
	public static Node findSubNode(Node node, int beginline, int endline, String subnodecode) {
		Node[] subnode = new Node[1];
		findSubNode_(node, beginline, endline, subnodecode, subnode);
		return subnode[0];
	}
	
	public static void removeComments(Node node) {
		node = node.removeComment();
		for (Comment comment : node.getAllContainedComments()) {
            comment.remove();
        }
		for (Node child : node.getChildNodes()) {
			removeComments(child);
		}
		// return node.removeComment();
	}
	
	public static JNode calculateDepth(Node node, int depth) {
		JNode JNode = new JNode(node, depth);
		for (Node child : node.getChildNodes()) {
			JNode mychild = calculateDepth(child, depth + 1);
			JNode.addChild(mychild);
		}
		return JNode;
	}
	
	public static ArrayList<String> analysisImports(JNode myroot){
		ArrayList<String> imports = new ArrayList<String>();
		ArrayList<JNode> dfs = myroot.DFS();
		for (JNode JNode : dfs) {
			if (JNode.classEquals(ImportDeclaration.class)) {
				imports.add(JNode.getChildrenByTpye(Name.class).get(0).getValue());
			}
		}
		return imports;
	}
	
	public static String analysisPackages(JNode myroot) {
		ArrayList<JNode> dfs = myroot.DFS();
		for (JNode JNode : dfs) {
			if (JNode.classEquals(PackageDeclaration.class)) {
				 return JNode.getChildrenByTpye(Name.class).get(0).getValue();
			}
		}
		return null;
	}

    public static void findContext_(Node node, int beginline, int endline, String code, Node[] subnode) {
		if (containLines(node, beginline, endline, code)) {
			if (node.getClass().equals(MethodDeclaration.class) || node.getClass().equals(ConstructorDeclaration.class))
				subnode[1] = node;
			subnode[0] = node;
			List<Node> childs = node.getChildNodes();
			for (Node child : childs) {
				findContext_(child, beginline, endline, code, subnode);
			}
		}
	}
	
	public static Node findContextNode(Node node, int beginline, int endline, String code) {
		Node[] subnode = {null, null};
		findContext_(node, beginline, endline, code, subnode);
		return subnode[1];
	}

    public static Node findContextNode(String filepath, int beginline, int endline) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		String line;
		int cnt = 0;
		String code = "";
		while ((line = br.readLine()) != null) {
			cnt += 1;
			if (beginline <= cnt && cnt < endline)
				code += line.trim() + " ";
			if (cnt >= endline)
				break;
		}
		br.close();
		code = code.trim();
		
		try {
			Node root = Parser.parseFile(filepath);
			return findContextNode(root, beginline, endline, code);
		} catch (TokenMgrError e) {
			System.out.println(e.getStackTrace());
			return null;
		}
	}

    public static String findContext(String filepath, int beginline, int endline) throws IOException {
		BufferedReader br = new BufferedReader(new FileReader(new File(filepath)));
		String line;
		int cnt = 0;
		String code = "";
		while ((line = br.readLine()) != null) {
			cnt += 1;
			if (beginline <= cnt && cnt < endline)
				code += line.trim() + " ";
			if (cnt >= endline)
				break;
		}
		br.close();
		code = code.trim();
		
		try {
			Node root = Parser.parseFile(filepath);
			Node context =  findContextNode(root, beginline, endline, code);
			return context == null ? "" : context.toString();
		} catch (TokenMgrError e) {
			return "";
		}
	}
	
	public static void main(String[] args) throws Exception {
		Node root = Parser.parseFile("D:\\java-eclipse-workspace\\deeprepair-javaparser\\src\\test\\java\\jiang719\\test\\Test1.java");
		JNode myroot = Parser.calculateDepth(root, 0);
		for (JNode JNode : myroot.DFS()) {
			if (JNode.node instanceof NodeWithIdentifier) {
				System.out.println(((NodeWithIdentifier<?>) JNode.node).getIdentifier());
			}
		}
		
	}
}
