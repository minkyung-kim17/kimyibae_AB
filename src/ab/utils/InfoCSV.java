package ab.utils;

import java.util.ArrayList;

import java.io.BufferedWriter;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.nio.charset.Charset;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;

import ab.vision.GameStateExtractor.GameState;

public class InfoCSV {
	
	public static ArrayList<ArrayList<String>> info_set = new ArrayList<ArrayList<String>>(); // 게임 한번(level 1-21) 할때마다 생기는 정보 저장
	public static ArrayList<ArrayList<String>> info_set_level = new ArrayList<ArrayList<String>>(); // 게임 한번(level 1-21) 할때마다 생기는 정보 저장
	static ArrayList<String> info_set_col = new ArrayList<String>(); 

	public static ArrayList<String> info_oneshot = new ArrayList<String>();
	
	//쓰잘데기 없는거 -_-;;;
	//추가가 계속됨 변수 범위? 같은 문제이듯?
	public static ArrayList<ArrayList<String>> add_info_set_col (ArrayList<ArrayList<String>> info_set){
		info_set_col.add("LEVEL"); 
		info_set_col.add("ANGLE");
		info_set_col.add("SCORE");
		info_set_col.add("STATE");
		info_set_col.add("PIGS");
		info_set.add(info_set_col);
		return info_set;
	}
	
	//굳이 함수를 쓰지 않아도 됨
	//.add()도 가능! String을 위한 형변환을 보여줌
	public static ArrayList<String> add_info_level (ArrayList<String> info_oneshot, int level) {
		info_oneshot.add(String.valueOf(level));
		return info_oneshot;
	}
	
	public static ArrayList<String> add_info_angle (ArrayList<String> info_oneshot, double angle) {
		info_oneshot.add(Double.toString(angle));
		return info_oneshot;
	}
	
	public static ArrayList<String> add_info_score (ArrayList<String> info_oneshot, int score) {
		info_oneshot.add(String.valueOf(score));
		return info_oneshot;
	}
	
	public static ArrayList<String> add_info_state (ArrayList<String> info_oneshot, GameState state) {
		info_oneshot.add(state.toString());
		return info_oneshot;
	}

	//ArrayList 추가도 .add로 충분히 가능
	public static ArrayList<ArrayList<String>> add_info_oneshot (ArrayList<ArrayList<String>> info_set, ArrayList<String> info_oneshot) {
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
		System.out.println("");
	}

	public static void print_info(ArrayList<String> info_oneshot) {
		ArrayList<String> list_one = info_oneshot;
		for (String data:list_one) {
			System.out.print(data+" ");
		}
		System.out.println("");
	}
	
	//밖에서 가져다 쓰는 것
	public static void writecsv(ArrayList<ArrayList<String>> info_set, String filepath) {
//		String pwd = System.getProperty("user.dir");
//		String filepath = pwd+"/info.csv";
//		System.out.println(filepath);
		BufferedWriter bufWriter = null;
		try {
			bufWriter = Files.newBufferedWriter(Paths.get(filepath));
			
			for (ArrayList<String> newLine : info_set) {
				ArrayList<String> list = newLine;
				for(String data : list) {
					bufWriter.write(data);
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
	
	public static void main(String args[]) {
		//list 만들기, 2차원 배열 --> info_set
		//           2차원 배열의 row --> info_oneshot 
				
		info_set = add_info_set_col(info_set);
		
		info_oneshot = add_info_score(info_oneshot, 300);
		info_oneshot = add_info_score(info_oneshot, 500);
		
		info_set = add_info_oneshot(info_set, info_oneshot);
		
		print_infoset(info_set);
		
		String pwd = System.getProperty("user.dir");
		String filepath = pwd+"/info.csv";
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
