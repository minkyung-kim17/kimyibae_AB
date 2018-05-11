package ab.utils;

import java.util.ArrayList;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

public class InfoCSV {
	
	static ArrayList<ArrayList<String>> info_set = new ArrayList<ArrayList<String>>(); // 게임 한번(level 1-21) 할때마다 생기는 정보 저장
	static ArrayList<String> info_set_col = new ArrayList<String>(); 
	static ArrayList<String> info_oneshot = new ArrayList<String>();
	
	private static ArrayList<ArrayList<String>> add_info_set_col (ArrayList<ArrayList<String>> info_set){
		info_set_col.add("SCORE"); 
		info_set_col.add("LEVEL");
		info_set.add(info_set_col);
		return info_set;
	}
	
	private static ArrayList<String> add_info_score (ArrayList<String> info_oneshot, int score) {
		info_oneshot.add(String.valueOf(score));
		return info_oneshot;
	}
	
	private static ArrayList<ArrayList<String>> add_info_oneshot (ArrayList<ArrayList<String>> info_set, ArrayList<String> info_oneshot) {
		info_set.add(info_oneshot);
		return info_set;
	}
	
	public static void print_infoset(ArrayList<ArrayList<String>> info_set) {
		for (ArrayList<String> newLine : info_set) {
			ArrayList<String> list_set = newLine;
			System.out.println("");
			for(String data: list_set) {
				System.out.print(data+" ");
			}
			System.out.println("");
		}
	}

	public static void print_info(ArrayList<String> info_oneshot) {
		ArrayList<String> list_one = info_oneshot;
		for (String data:list_one) {
			System.out.print(data+" ");
		}
		System.out.println("");
	}
	
	public static void main(String args[]) {
		//list 만들기, 2차원 배열 --> info_set
		//           2차원 배열의 row --> info_oneshot 
				
		info_set = add_info_set_col(info_set);
		
//		print_infoset(info_set);
		
		info_oneshot = add_info_score(info_oneshot, 300);
		info_oneshot = add_info_score(info_oneshot, 500);

//		print_info(info_oneshot);
		
		info_set = add_info_oneshot(info_set, info_oneshot);
		
		print_infoset(info_set);
		
		String pwd = System.getProperty("user.dir");
		String filepath = pwd+"/info.csv";
//		System.out.println(filepath);
		BufferedWriter bufWriter = null;
		try {
			bufWriter = Files.newBufferedWriter(Paths.get(filepath));
			
			for (ArrayList<String> newLine : info_set) {
				ArrayList<String> list = newLine;
				for(String data : list) {
					bufWriter.write(data);
					System.out.println(data);
					bufWriter.write(",");
				}
				bufWriter.newLine();
			}
			System.out.println("save");
		} catch (IOException e) {
			e.printStackTrace();
		} finally {
			try {
				if(bufWriter != null) {
					bufWriter.close();
				}
			} catch(IOException e) {
				e.printStackTrace();
			}
		}
	}
}
