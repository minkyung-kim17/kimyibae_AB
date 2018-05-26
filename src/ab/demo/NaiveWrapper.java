package ab.demo;

import py4j.GatewayServer;

import java.util.Arrays;
import java.io.IOException;
import java.io.File;
import java.awt.image.BufferedImage;
import java.awt.Rectangle;
import java.awt.Point;
import java.awt.Graphics2D;
import java.util.List;
import javax.imageio.ImageIO;

import ab.planner.abTrajectory;
import ab.utils.GameImageRecorder;
import ab.vision.ShowSeg;
import ab.vision.GameStateExtractor;
import ab.vision.GameStateExtractor.GameState;
import ab.vision.VisionRealShape;
import ab.vision.Vision;
import ab.vision.ABObject;
/*****************************************************************************
 ** ANGRYBIRDS AI AGENT FRAMEWORK
 ** Copyright (c) 2014, XiaoYu (Gary) Ge, Jochen Renz,Stephen Gould,
 **  Sahan Abeyasinghe,Jim Keys,  Andrew Wang, Peng Zhang
 ** All rights reserved.
**This work is licensed under the terms of the GNU Affero General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.
**To view a copy of this license, visit http://www.gnu.org/licenses/
 *****************************************************************************/

public class NaiveWrapper {
	// the entry of the software.

	private static GameStateExtractor gse;

	public NaiveWrapper() {
		gse = new GameStateExtractor();
	}

	public void sayHello() {
		java.lang.System.out.println("Hello World!");
	}

	public void getStateTest(String path){
		try{
			//File file = new File("/home/jongchan/Projects/AIBIRD/LEARNERSHIGH/AB_Wrap/screenshot.png");
			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);
			int score = gse.getScoreInGame(screenshot);
			System.out.println("score:"+score);
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
		}
	}

	public int getScoreInGame(String path){
		try{
			//File file = new File("/home/jongchan/Projects/AIBIRD/LEARNERSHIGH/AB_Wrap/screenshot.png");
			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);
			int score = gse.getScoreInGame(screenshot);
			System.out.println("score:"+score);
			return score;
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return -1;
		}
	}
	public int getScoreEndGame(String path){
		try{
			//File file = new File("/home/jongchan/Projects/AIBIRD/LEARNERSHIGH/AB_Wrap/screenshot.png");
			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);
			int score = gse.getScoreEndGame(screenshot);
			System.out.println("score:"+score);
			return score;
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return -1;
		}
	}
	public int getScoreGame(String path, String game_state){
		try{
			//File file = new File("/home/jongchan/Projects/AIBIRD/LEARNERSHIGH/AB_Wrap/screenshot.png");
			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);
			int score = 0;
			if (game_state == "PLAYING")
				score = gse.getScoreInGame(screenshot);
			else
				score = gse.getScoreEndGame(screenshot);

			System.out.println("score:"+score);
			return score;
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return -1;
		}
	}

	public int[] findSlingshot(String path){
		try{

			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);

			Vision vision = new Vision(screenshot);
			Rectangle sling = vision.findSlingshotMBR();
			int[] x_y = new int[] {-1, -1, -1, -1};
			if (sling==null)
				return x_y;
			else{
				//double X_OFFSET = 0.5;
				//double Y_OFFSET = 0.65;
				//Point p = new Point((int)(sling.x + X_OFFSET * sling.width), (int)(sling.y + Y_OFFSET * sling.width));
				x_y[0] = sling.x;
				x_y[1] = sling.y;
				x_y[2] = sling.width;
				x_y[3] = sling.height;
				System.out.println("sling found at rect ("+sling.x+","+sling.y+","+sling.width+","+sling.height+")");
				return x_y;
			}
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return new int[] {-1, -1, -1, -1};
		}
	}

	public boolean saveSegWithPath(String path, String save_path){
		try{

			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);

			VisionRealShape vision = new VisionRealShape(screenshot);
			vision.findObjects();
			vision.findPigs();
			vision.findHills();
			vision.findBirds();
			vision.findSling();
			vision.findTrajectory();


			//vision.drawObjects(screenshot, true);
			BufferedImage seg = vision.drawObjectsInCanvas(true);

			File outputFile = new File(save_path);
			ImageIO.write(seg, "png", outputFile);

			return true;
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return false;
		}
	}

	public int[][] findBlocks(String path, boolean MBR){
		try{
			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);
			if (MBR)
				return findBlocksMBR(screenshot);
			else
				return findBlocksRealShape(screenshot);
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return null;
		}
	}

	public int[][] findBlocksMBR(BufferedImage screenshot){
		Vision vision = new Vision(screenshot);
		List<ABObject> blocks_abobject = vision.findBlocksMBR();
		int[][] blocks = new int[blocks_abobject.size()][5];
		for (int i = 0; i<blocks_abobject.size(); i++){
			ABObject block_abobject = blocks_abobject.get(i);
			blocks[i][0] = block_abobject.type.id;
			blocks[i][1] = block_abobject.x;
			blocks[i][2] = block_abobject.y;
			blocks[i][3] = block_abobject.width;
			blocks[i][4] = block_abobject.height;
		}
		return blocks;
	}

	public int[][] findBlocksRealShape(BufferedImage screenshot){
		Vision vision = new Vision(screenshot);
		List<ABObject> blocks_abobject = vision.findBlocksRealShape();
		int[][] blocks = new int[blocks_abobject.size()][5];
		return blocks;
	}

	public int[][] findPigs(String path, boolean MBR){
		try{
			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);
			if (MBR)
				return findPigsMBR(screenshot);
			else
				return findPigsRealShape(screenshot);
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return null;
		}
	}

	public int[][] findPigsMBR(BufferedImage screenshot){
		Vision vision = new Vision(screenshot);
		List<ABObject> pigs_abobject = vision.findPigsMBR();
		int[][] pigs = new int[pigs_abobject.size()][5];
		for (int i = 0; i<pigs_abobject.size(); i++){
			ABObject pig_abobject = pigs_abobject.get(i);
			pigs[i][0] = pig_abobject.type.id;
			pigs[i][1] = pig_abobject.x;
			pigs[i][2] = pig_abobject.y;
			pigs[i][3] = pig_abobject.width;
			pigs[i][4] = pig_abobject.height;
		}
		return pigs;
	}

	public int[][] findPigsRealShape(BufferedImage screenshot){
		Vision vision = new Vision(screenshot);
		List<ABObject> pigs_abobject = vision.findPigsRealShape();
		int[][] pigs = new int[pigs_abobject.size()][5];
		return pigs;
	}

	public int[][] findBirds(String path, boolean MBR){
		try{
			File file = new File(path);
			BufferedImage screenshot = ImageIO.read(file);
			if (MBR)
				return findBirdsMBR(screenshot);
			else
				return findBirdsRealShape(screenshot);
		}catch (IOException e){
			System.err.println("failed to load resources");
			e.printStackTrace();
			return null;
		}
	}

	public int[][] findBirdsRealShape(BufferedImage screenshot){
		Vision vision = new Vision(screenshot);
		List<ABObject> birds_abobject = vision.findBirdsRealShape();
		int[][] birds = new int[birds_abobject.size()][5];
		return birds;
	}

	public int[][] findBirdsMBR(BufferedImage screenshot){
		Vision vision = new Vision(screenshot);
		List<ABObject> birds_abobject = vision.findBirdsMBR();
		int[][] birds = new int[birds_abobject.size()][5];
		for (int i = 0; i<birds_abobject.size(); i++){
			ABObject bird_abobject = birds_abobject.get(i);
			birds[i][0] = bird_abobject.type.id;
			birds[i][1] = bird_abobject.x;
			birds[i][2] = bird_abobject.y;
			birds[i][3] = bird_abobject.width;
			birds[i][4] = bird_abobject.height;
		}
		return birds;
	}

	public static void main(String args[])
	{
		 GatewayServer gatewayServer = new GatewayServer(new NaiveWrapper(), 20001);
//		GatewayServer gatewayServer = new GatewayServer(new NaiveWrapper());
		gatewayServer.start();
		System.out.println("Running GatewayServer...");
	}
}
