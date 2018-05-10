package ab.demo;

import java.io.BufferedWriter;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.OutputStreamWriter;

public class CSVWrite {
	public static void main(String filename, String enc) {
		//Set save_path, encoding format
//		String save_path = System.getProperty("user.dir");
//		String csvFileName = save_path + "/info.csv";
//		String enc = new java.io.OutputStreamWriter(System.out).getEncoding();

		//Set contents
//		String col_name = "LEVEL, RP, ANGLE, SCORE, STATE";
//		String col_name = "LEVEL, RP, ANGLE";
		String col_name = "LEVEL";
		info_add(col_name, filename, enc);
	}
	
	public static void info_add(String s, String filename, String enc) {
		try{
			BufferedWriter writer = new BufferedWriter(new OutputStreamWriter(new FileOutputStream(filename, true), enc));
			writer.write(s+"\n");
			writer.close();
			
		} catch (IOException e) {
			e.printStackTrace();
		}
	}
}
