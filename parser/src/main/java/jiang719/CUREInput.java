package jiang719;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Set;

import com.fasterxml.jackson.core.JsonGenerationException;
import com.fasterxml.jackson.databind.JsonMappingException;
import com.fasterxml.jackson.databind.ObjectMapper;


public class CUREInput {
	
	public static HashMap<String, Object> result = new HashMap<String, Object>();
	
	public static void findContext(String inputPath, int beginline, int endline) throws Exception {
		result.put("context", Parser.findContext(inputPath, beginline, endline));
	}

	public static void findBuggyLine(String inputPath, int beginline, int endline) throws Exception{
		BufferedReader br = new BufferedReader(new FileReader(new File(inputPath)));
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
		result.put("buggy line", code);
	}
	
	static int findSubList(String[] L1, String[] L2) {
		for (int i = 0; i < L1.length; i += 1) {
			boolean find = true;
			for (int j = 0; j < L2.length; j += 1) {
				if (! L1[i + j].equals(L2[j])) {
					find = false;
					break;
				}
			}
			if (find == true)
				return i;
		}
		return -1;
	}
	
	public static void findIdentifiers(String inputPath, int beginline, int endline) throws Exception {
		JNode root = Parser.calculateDepth(Parser.parseFile(inputPath), 0);
		
		Set<String> identifiers = new HashSet<String>();
		
		ArrayList<String> imports = new ArrayList<String>();
		String path = inputPath;
		path = path.substring(0, path.lastIndexOf("/"));
		// package identifiers
		identifiers.addAll(IdentifierParser.analysisPackageIdentifiers(path));
		
		String[] base = path.split("/");
		int baseIndex = findSubList(base, Parser.analysisPackages(root).split("\\."));
		String basepath = "";
		for (int i = 0; i < baseIndex; i += 1)
			basepath += base[i] + "/";
		// imported identifiers
		for (String str : Parser.analysisImports(root)) {
			if (str.length() >= 5 && str.substring(0, 5).equals("java.")) {
				imports.add(str);
				continue;
			}
			
			path = basepath;
			for (String dir : str.split("\\.")) {
				path += "/" + dir;
				if (dir.charAt(0) >= 'A' && dir.charAt(0) <= 'Z')
					break;
			}
			if (path.length() < 5 || ! path.substring(0, path.length() - 5).equals(".java"))
				path += ".java";
			File file = new File(path);
			if (file.exists()) {
				identifiers.addAll(IdentifierParser.analysisImportedFile(path));
			}
		}
		
		// in-scope identifiers
		Map<Integer, Set<String>> identifierScope = IdentifierParser.analysisIdentifierScope(root);
		for (Integer k : identifierScope.keySet()) {
			if (k >= beginline && k < endline)
				identifiers.addAll(identifierScope.get(k));
		}
		
		result.put("imports", imports);
		result.put("identifiers", identifiers);
	};
	
	public static void write(String outputPath) throws JsonGenerationException, JsonMappingException, IOException {
		ObjectMapper writer = new ObjectMapper();
		writer.writeValue(new File(outputPath), result);
	}
	
	public static void main(String[] args) throws Exception {
		if (args.length == 4) {
			String inputPath = args[0];
			String outputPath = args[1];
			int beginline = Integer.parseInt(args[2]);
			int endline = Integer.parseInt(args[3]);
			
			findBuggyLine(inputPath, beginline, endline);
			findContext(inputPath, beginline, endline);
			findIdentifiers(inputPath, beginline, endline);
			write(outputPath);
		}
        else if (args.length == 0){
            String inputPath = "/local/n44jiang/CURE/parser/src/test/java/jiang719/Test.java";
            String outputPath = "output.json";
            int beginline = 10;
            int endline = 11;
			findBuggyLine(inputPath, beginline, endline);
            findContext(inputPath, beginline, endline);
            findIdentifiers(inputPath, beginline, endline);
            write(outputPath);
        }
	}

}
