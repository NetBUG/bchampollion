/**
 * JChampollion is an implementation of the Champollion algorithm,
 * as detailed in the 1996 paper by Smadja, McKeown and Hatzivassiloglou 
 * entitled "Translating Collocations for Bilingual Lexicons: A 
 * Statistical Approach"
 */
package jchampollion;

import org.apache.lucene.analysis.Analyzer;

/**
 * @author Adam Goforth
 *
 */
public class JChampollion {

	private Corpus corpus;
	private String sourcefilename;
	private String targetfilename;
	private String collocation;
	private boolean doIndex;
	private boolean showHelp;
	private double Td;
	private int Tf;
	
	/**
	 * main - creates an instance of a JChampollion and tells it to handle
	 * the command line arguments
	 * 
	 * @param args The command line arguments.
	 */
	public static void main(String[] args) {
		JChampollion champollion = new JChampollion();
		champollion.parseProgArgs(args);
	}
	
	/**
	 * Constructor
	 * @param filename 	The name of the file with the corpus to be analyzed.
	 */
	public JChampollion(){
		doIndex = false;
		showHelp = false;
		sourcefilename = "";
		targetfilename = "";
		collocation = "";
		Td = 0.1;
		Tf = 5;
	}
	
	/**
	 * Translates the given collocation
	 *
	 */
	public void translate(String collocation){
		System.out.println(corpus.getTranslation(collocation, Tf, Td));
	}
	
	/**
	 * Rebuilds the index.   Run once each time the corpus changes.
	 *
	 */
	private void IndexCorpus(){
		corpus.buildIndices();
	}
	
	/**
	 * showHelp
	 * Prints the command line syntax and quits.
	 */
	public void showHelp(){
		System.out.println("JChampollion v0.1");
		System.out.println("Usage: java -jar jchamp.jar -source filename -target filename [-index] [-Td threshold] [-Tf threshold] -co collocation");
		System.out.println("Example: java -jar jchamp.jar -source en.txt -target de.txt -co \"member states\"");
		System.out.println("Arguments:");
		System.out.println("-source\t\tThe path to the English language corpus, either a file or a directory.");
		System.out.println("-target\t\tThe path to the German language corpus, either a file or a directory.");
		System.out.println("-co\t\tThe collocation to be translated.  Must be enclosed in quotes.");
		System.out.println("-index\t\tRe-build the index of the corpora.  This should be run once each time");
		System.out.println("\t\t\tthe corpus changes.");
		System.out.println("-Td\t\t(Optional)The dice threshold.  Defaults to 0.1");
		System.out.println("-Tf\t\t(Optional)The frequency threshold.  Defaults to 5");
		System.out.println("-h\t\tPrints this help message.");
	}
	
	/**
	 * Set class' variables based on command line arguments.
	 * @param args	Command line arguments.
	 */
	public void parseProgArgs(String[] args){
		for (int i = 0; i < args.length; i++) {
			if (args[i].equals("-source")) {
				sourcefilename = args[i + 1];
			} else if (args[i].equals("-target")) {
				targetfilename = args[i + 1];
			} else if (args[i].equals("-index")) {
				doIndex = true;
			} else if (args[i].equals("-Td")) {
				Td = Double.parseDouble(args[i+1]);
			} else if (args[i].equals("-Tf")) {
				Tf = Integer.parseInt(args[i+1]);
			} else if (args[i].equals("-co")){
				collocation = args[i + 1];
			} else if (args[i].equals("-h") || args[i].equals("-help") || args[i].equals("--help")){
				showHelp = true;
			}
		}
		
		if (showHelp){
			showHelp();
			System.exit(0);
		}
		
		// Make sure source, target and collocation were given
		if (!sourcefilename.equals("") && !targetfilename.equals("") && !collocation.equals("")){
			corpus = new Corpus(sourcefilename, targetfilename);
			
			if (doIndex){
				IndexCorpus();
			}
			
			translate(collocation);

		} else {
			showHelp();
			System.exit(0);
		}
	}
}
