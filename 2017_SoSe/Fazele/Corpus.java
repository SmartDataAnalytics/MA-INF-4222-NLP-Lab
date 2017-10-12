package com.company;
import java.util.*;
import java.io.*;
import java.lang.*;
import java.util.regex.Pattern;


public class Corpus  {

    private static final String REGEXSINGLESPACE = "^\\s$";
    public static final Pattern pSingleSpace = Pattern.compile(REGEXSINGLESPACE);
    private static final String REGEXEMPTY = "^$";
    public static final Pattern pEmpty = Pattern.compile(REGEXEMPTY);

    public static List<String> MarujoFileSelector_InputDocument(){
        File folder = new File("/Users/fazeletavakoli/IdeaProjects/stanford-corenlp/testData");
        File[] listOfFiles = folder.listFiles();
        int counter = 1;
        List <String> desiredFiles = new ArrayList<>();
        boolean isTrue = false; //for ignoring a .DS_Store file in the folder, which is located at the beginning of the folder

        for (File file : listOfFiles) {
            if (!file.getPath().contains(".DS_Store")) {
                if (counter % 4 == 0) {

                    String s = file.getPath();
                    desiredFiles.add(s);
                }
                counter ++;
            }
        }
        return desiredFiles;
    }

    public static List<String> MarujoFileSelector_Test(){

        File folder = new File("/Users/fazeletavakoli/IdeaProjects/stanford-corenlp/testData");
        File[] listOfFiles = folder.listFiles();
        int counter = 1;
        List <String> desiredFiles = new ArrayList<>();
        boolean isTrue = false; //for ignoring a .DS_Store file in the folder, which is located at the beginning of the folder

        for (File file : listOfFiles) {

                if(file.getName().contains(".key")){
                    String s = file.getPath();
                    desiredFiles.add(s);
                }
        }

        return desiredFiles;
    }

    //this method converts .key files to .txt files
    public static String convertKeyToTxt(File inputFile) {
        int index = inputFile.getPath().indexOf(".");
        String primaryName = inputFile.getPath().substring(0,index);
        //use file.renameTo() to rename the file
        inputFile.renameTo(new File(primaryName +"."+"txt"));
        return inputFile.getPath();

    }

    public static void writeInFile_version2(String fileInput,File inputFile) {
        BufferedWriter bw = null;
        try {
            FileWriter fw = new FileWriter(inputFile, true);
            bw = new BufferedWriter(fw);
            bw.write(fileInput);
            bw.write("\r\n");
            System.out.println("File written Successfully");

        } catch (IOException ioe) {
            ioe.printStackTrace();
        } finally {
            try {
                if (bw != null)
                    bw.close();
            } catch (Exception ex) {
                System.out.println("Error in closing the BufferedWriter" + ex);
            }

        }
    }

    public static void FileCleaner_version2(File inputFile){
        // empty the current content
        try {
            FileWriter fw = new FileWriter(inputFile);
            fw.write("");
            fw.close();
        }catch(IOException ioe){
            ioe.printStackTrace();
        }
    }

    public static void computeAvgPrecRecal(String filePath){
        int index=0;
        int index1=0;
        Double precisionDouble =0.0;
        Double recallDouble = 0.0;
        Double counter = 0.0;
        try {
            String line = "";
            BufferedReader br = null;
            InputStream inputStream = new FileInputStream(filePath);
            br = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));
            while ((line = br.readLine()) != null) {
                if (!line.equals("")) {
                    if (line.contains("precision")) {
                        index = line.indexOf("precision");
                    }
                    if (line.contains("Recall")) {
                        index1 = line.indexOf("Recall");
                    }
                    String precisionStr = line.substring(index + 11, index1 - 2);
                    precisionDouble += Double.parseDouble(precisionStr);
                    String recallStr = line.substring(index1 + 8);
                    recallDouble += Double.parseDouble(recallStr);
                    counter++;
                }
            }
            Double avgPrecision = precisionDouble/counter;
            Double avgRecall = recallDouble/counter;
            System.out.println("average precision: "+ avgPrecision +"\n" + "average recall: " + avgRecall);
        }catch (Exception e){

        }
    }

}

