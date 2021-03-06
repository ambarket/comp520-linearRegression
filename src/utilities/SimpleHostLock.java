package utilities;


import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.io.PrintWriter;
import java.net.InetAddress;

public class SimpleHostLock {
	public static boolean checkAndClaimHostLock(String hostLockFilePath) {
		return true;
		/*
		File hostLock;
		if ((hostLock = new File(hostLockFilePath)).exists()) {
			try {
				BufferedReader br = new BufferedReader(new FileReader(hostLock));
				String hostName = br.readLine();
				br.close();
				if (hostName == null) {
					BufferedWriter bw = new BufferedWriter(new PrintWriter(hostLock));
					bw.write(InetAddress.getLocalHost().getHostName() + "\n");
					bw.flush();
					bw.close();
					return true;
				}
				return (hostName).equals(InetAddress.getLocalHost().getHostName());
			} catch (FileNotFoundException e) {
				System.out.println("Host lock file exists but file not found. Makes no sense.");
			} catch (IOException e) {
				e.printStackTrace();
			}

		} else {
			try {
				BufferedWriter bw = new BufferedWriter(new PrintWriter(hostLock));
				bw.write(InetAddress.getLocalHost().getHostName() + "\n");
				bw.flush();
				bw.close();
				return true;
			} catch (IOException e) {
				e.printStackTrace();
			} 
		}
		System.out.println("ERROR Shouldnt reach here in checkAndClaimHostLock");
		return false;
					*/
	}
	
	public static boolean checkDoneLock(String doneLockFilePath) {
		return new File(doneLockFilePath).exists();
	}
	
	public static boolean writeDoneLock(String doneLockFilePath) {
		File doneLock;
		doneLock = new File(doneLockFilePath);
		try {
			BufferedWriter bw = new BufferedWriter(new PrintWriter(doneLock));
			bw.write(InetAddress.getLocalHost().getHostName() + "\n");
			bw.flush();
			bw.close();
			return true;
		} catch (IOException e) {
			e.printStackTrace();
		} 
		System.out.println("ERROR Shouldnt reach here in writeDoneLock");
		return false;
	}
}
