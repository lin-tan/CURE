package jiang719;

import java.io.File;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Set;

import com.github.javaparser.JavaParser;
import com.github.javaparser.Range;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Modifier;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.body.ConstructorDeclaration;
import com.github.javaparser.ast.body.EnumConstantDeclaration;
import com.github.javaparser.ast.body.EnumDeclaration;
import com.github.javaparser.ast.body.FieldDeclaration;
import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.body.Parameter;
import com.github.javaparser.ast.body.VariableDeclarator;
import com.github.javaparser.ast.expr.LambdaExpr;
import com.github.javaparser.ast.expr.SimpleName;
import com.github.javaparser.ast.expr.VariableDeclarationExpr;
import com.github.javaparser.ast.stmt.BlockStmt;
import com.github.javaparser.ast.stmt.CatchClause;
import com.github.javaparser.ast.stmt.ExpressionStmt;
import com.github.javaparser.ast.stmt.ForEachStmt;
import com.github.javaparser.ast.stmt.ForStmt;
import com.github.javaparser.ast.stmt.LocalClassDeclarationStmt;
import com.github.javaparser.ast.stmt.SwitchEntry;
import com.github.javaparser.ast.stmt.SwitchStmt;


public class IdentifierParser extends Parser {
	
	public static JavaParser parser = new JavaParser();
	
	public static Set<String> analysisPackageIdentifiers(String dir) {
		Set<String> identifiers = new HashSet<String>();
		File[] files = (new File(dir)).listFiles();
		for (File file : files) {
			String filename = file.getName();
			if (filename.length() < 5 || ! filename.substring(filename.length() - 5).equals(".java"))
				continue;
			try {
				JNode myroot = calculateDepth(parseFile(file), 0);
				ArrayList<JNode> dfs = myroot.DFS();
				for (JNode JNode : dfs) {
					
					if (JNode.classEquals(ClassOrInterfaceDeclaration.class)
							|| JNode.classEquals(EnumDeclaration.class))
						analysisClassOrInterfaceOrEnumDeclaration(JNode, identifiers);
				}
			}
			catch (Exception e) {
				System.out.println(filename);
			}
		}
		return identifiers;
	}
	
	public static Set<String> analysisImportedFile(String file){
		Set<String> identifiers = new HashSet<String>();
		try {
			JNode myroot = calculateDepth(parseFile(file), 0);
			ArrayList<JNode> dfs = myroot.DFS();
			for (JNode JNode : dfs) {
				if (JNode.classEquals(ClassOrInterfaceDeclaration.class))
					analysisClassOrInterfaceOrEnumDeclaration(JNode, identifiers);
			}
		}
		catch (Exception e) {
			System.out.println(file);
		}
		return identifiers;
	}
	
	public static HashMap<Integer, Set<String>> analysisIdentifierScope(JNode myroot){
		HashMap<Integer, Set<String>> identifierScope = new HashMap<Integer, Set<String>>();
		ArrayList<JNode> dfs = myroot.DFS();
		
		for (JNode JNode : dfs) {
			if (JNode.classEquals(ClassOrInterfaceDeclaration.class)) {
				analysisClassOrInterfaceOrEnumDeclaration(JNode, identifierScope);
			}
			else if (JNode.classEquals(VariableDeclarationExpr.class)) {
				analysisVariableDeclarationExpr(JNode, identifierScope);
			}
			else if (JNode.classEquals(Parameter.class)) {
				analysisParameter(JNode, identifierScope);
			}
		}
		
		return identifierScope;
	}
	
	public static void addIdentifier(String identifier, Range range, 
			HashMap<Integer, Set<String>> identifierScope) {
		int begin = range.begin.line;
		int end = range.end.line;
		for (int i = begin; i <= end; i += 1) {
			if (! identifierScope.containsKey(i))
				identifierScope.put(i, new HashSet<String>());
			identifierScope.get(i).add(identifier);
		}
	}
	
	public static void analysisParameter(JNode JNode, HashMap<Integer, Set<String>> identifierScope) {
		JNode father = JNode.father;
		JNode scope = null;
		if (father.classEquals(CatchClause.class) || father.classEquals(MethodDeclaration.class) || 
				father.classEquals(ConstructorDeclaration.class) || father.classEquals(LambdaExpr.class)) {
			scope = father;
		}
		else {
			System.out.println("Invalid father node for Parameter:" + 
					father.getClass());
			assert(false);
		}
		String identifier = JNode.getChildrenByTpye(SimpleName.class).get(0).getValue();
		addIdentifier(identifier, new Range(JNode.getRange().begin, scope.getRange().end), 
				identifierScope);
	}
	
	public static void analysisVariableDeclarationExpr(JNode JNode, 
			HashMap<Integer, Set<String>> identifierScope) {
		JNode father = JNode.father;
		JNode scope = null;
		if (father.classEquals(ExpressionStmt.class)) {
			if (father.father.classEquals(BlockStmt.class))
				scope = father.father;
			else if (father.father.classEquals(SwitchEntry.class)) {
				scope = father.father.father;
				assert(scope.classEquals(SwitchStmt.class));
			}
			else {
				System.out.println("Invalid father node for ExpressionStmt:" + 
						father.getClass());
				assert(false);
			}
		}
		else if (father.classEquals(ForStmt.class) || father.classEquals(ForEachStmt.class)) {
			scope = father;
		}
		else {
			System.out.println("Invalid father node for VariableDeclarationExpr:" + 
							father.getClass());
			assert(false);
		}
		
		for (JNode node : JNode.getChildrenByTpye(VariableDeclarator.class)) {
			String identifier = node.getChildrenByTpye(SimpleName.class).get(0).getValue();
			addIdentifier(identifier, new Range(JNode.getRange().begin, scope.getRange().end), 
					identifierScope);
		}
	}
	
	public static void analysisClassOrInterfaceOrEnumDeclaration(JNode JNode, 
			HashMap<Integer, Set<String>> identifierScope) {
		JNode father = JNode.father;
		JNode scope = null;
		if (father.classEquals(CompilationUnit.class) ||
				father.classEquals(ClassOrInterfaceDeclaration.class)) {
			scope = father;
		}
		else if(father.classEquals(LocalClassDeclarationStmt.class)) {
			scope = father.father;
			assert(scope.classEquals(BlockStmt.class));
		}
		else {
			System.out.println("Invalid father node for ClassOrInterfaceDeclaration:" + 
							father.getClass());
			assert(false);
		}
		
		String identifier = JNode.getChildrenByTpye(SimpleName.class).get(0).getValue();
		addIdentifier(identifier, scope.getRange(), identifierScope);
		
		if (JNode.getChildrenByTpye(EnumConstantDeclaration.class).size() > 0) {
			ArrayList<JNode> enms = JNode.getChildrenByTpye(EnumConstantDeclaration.class);
			for (JNode enm : enms) {
				String modifier = "public";
				if (enm.getChildrenByTpye(Modifier.class).size() > 0)
					modifier = enm.getChildrenByTpye(Modifier.class).get(0).getValue().trim();
				
				identifier = enm.getChildrenByTpye(SimpleName.class).get(0).getValue();
				if (modifier.equals("private")) {
					addIdentifier(identifier, JNode.getRange(), identifierScope);
				}
				else if(modifier.equals("public")) {
					addIdentifier(identifier, scope.getRange(), identifierScope);
				}
			}
		}
		
		if (JNode.getChildrenByTpye(FieldDeclaration.class).size() > 0) {
			ArrayList<JNode> fields = JNode.getChildrenByTpye(FieldDeclaration.class);
			for (JNode field : fields) {
				String modifier = "public";
				if (field.getChildrenByTpye(Modifier.class).size() > 0)
					modifier = field.getChildrenByTpye(Modifier.class).get(0).getValue().trim();
				for (JNode node : field.getChildrenByTpye(VariableDeclarator.class)) {
					identifier = node.getChildrenByTpye(SimpleName.class).get(0).getValue();
					if (modifier.equals("private")) {
						addIdentifier(identifier, JNode.getRange(), identifierScope);
					}
					else if(modifier.equals("public")) {
						addIdentifier(identifier, scope.getRange(), identifierScope);
					}
				}
			}
		}
		
		if (JNode.getChildrenByTpye(MethodDeclaration.class).size() > 0) {
			ArrayList<JNode> methods = JNode.getChildrenByTpye(MethodDeclaration.class);
			for (JNode method : methods) {
				String modifier = "public";
				if (method.getChildrenByTpye(Modifier.class).size() > 0)
					modifier = method.getChildrenByTpye(Modifier.class).get(0).getValue().trim();
				identifier = method.getChildrenByTpye(SimpleName.class).get(0).getValue();
				if (modifier.equals("private")) {
					addIdentifier(identifier, JNode.getRange(), identifierScope);
				}
				else if(modifier.equals("public")) {
					addIdentifier(identifier, scope.getRange(), identifierScope);
				}
			}
		}
		
	}
	
	public static void analysisClassOrInterfaceOrEnumDeclaration(JNode JNode, Set<String> packageIdentifier) {
		JNode father = JNode.father;
		if (! father.classEquals(CompilationUnit.class) &&
				! father.classEquals(ClassOrInterfaceDeclaration.class))
			return;
		
		String identifier = JNode.getChildrenByTpye(SimpleName.class).get(0).getValue();
		if (father.classEquals(CompilationUnit.class))
			packageIdentifier.add(identifier);
		else if (father.classEquals(ClassOrInterfaceDeclaration.class)) {
			String modifier = "public";
			if (JNode.getChildrenByTpye(Modifier.class).size() > 0)
				modifier = JNode.getChildrenByTpye(Modifier.class).get(0).getValue().trim();
			if (! modifier.equals("private"))
				packageIdentifier.add(identifier);
		}
		
		if (JNode.getChildrenByTpye(EnumConstantDeclaration.class).size() > 0) {
			ArrayList<JNode> enms = JNode.getChildrenByTpye(EnumConstantDeclaration.class);
			for (JNode enm : enms) {
				String modifier = "public";
				if (enm.getChildrenByTpye(Modifier.class).size() > 0)
					modifier = enm.getChildrenByTpye(Modifier.class).get(0).getValue().trim();
				
				identifier = enm.getChildrenByTpye(SimpleName.class).get(0).getValue();
				if (! modifier.equals("private")) {
					packageIdentifier.add(identifier);
				}
			}
		}
		
		if (JNode.getChildrenByTpye(FieldDeclaration.class).size() > 0) {
			ArrayList<JNode> fields = JNode.getChildrenByTpye(FieldDeclaration.class);
			for (JNode field : fields) {
				String modifier = "public";
				if (field.getChildrenByTpye(Modifier.class).size() > 0)
					modifier = field.getChildrenByTpye(Modifier.class).get(0).getValue().trim();
				
				for (JNode node : field.getChildrenByTpye(VariableDeclarator.class)) {
					identifier = node.getChildrenByTpye(SimpleName.class).get(0).getValue();
					if (! modifier.equals("private"))
						packageIdentifier.add(identifier);
				}
			}
		}
		
		if (JNode.getChildrenByTpye(MethodDeclaration.class).size() > 0) {
			ArrayList<JNode> methods = JNode.getChildrenByTpye(MethodDeclaration.class);
			for (JNode method : methods) {
				String modifier = "public";
				if (method.getChildrenByTpye(Modifier.class).size() > 0)
					modifier = method.getChildrenByTpye(Modifier.class).get(0).getValue().trim();
				
				identifier = method.getChildrenByTpye(SimpleName.class).get(0).getValue();
				if (! modifier.equals("private")) {
					packageIdentifier.add(identifier);
				}
			}
		}
	}
}
