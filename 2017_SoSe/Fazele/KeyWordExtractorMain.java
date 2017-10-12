package com.company;
import java.io.*;
import java.util.*;

public class KeyWordExtractorMain {
    public static void main(String[] args) {

        List<String> inputFilesPaths = new ArrayList<>(Corpus.MarujoFileSelector_InputDocument());
        List<String> testFilesPaths = new ArrayList<>(Corpus.MarujoFileSelector_Test());
        File testResultFile = new File("/Users/fazeletavakoli/IdeaProjects/stanford-corenlp/testResults.txt");  //output file
        Corpus.FileCleaner_version2(testResultFile);

        for (int fileNumber = 0; fileNumber < inputFilesPaths.size(); fileNumber++) {
            String filePath = inputFilesPaths.get(fileNumber);
            //String filePath = "/Users/fazeletavakoli/Desktop/KeywordInput_NLPLab/art_and_culture-20918624.txt";

            Spliter spliter = new Spliter();
            Graph graph = new Graph();
            String splittedArray[];
            List<Map<String, Map<String, Integer>>> kcoreNodesList = new ArrayList<>();
            boolean isTrue1 = false; //for checking if program enters while loop for second or ... time
            boolean isTrue2 = false; //for storing just KCores of the main graph
            List<Double> kCoreDensity = new ArrayList<>();


            graph.renew_G1();      //renewing Graph (for several files)
            graph.renew_stemmedtermlist(); //renewing stemmedTermList (for several files)

            try {
                String line = "";
                BufferedReader br = null;
                InputStream inputStream = new FileInputStream(filePath);
                br = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));
                while ((line = br.readLine()) != null) {
                    line = line.replaceAll("[\uFEFF]", "");
                    line = line.trim();
                    splittedArray = spliter.lineSplitter(line);
                    for (int i = 0; i < splittedArray.length; i++) {
                        splittedArray[i] = splittedArray[i].trim();
                        String sentence = splittedArray[i];
                        if (!Corpus.pSingleSpace.matcher(sentence).find() && !Corpus.pEmpty.matcher(sentence).find()) {
                            graph.setStemmedTermList_English(sentence);
                        }
                    }
                }

            } catch (IOException e) {
                e.printStackTrace();
            }

            SingletoneKCore singletoneKCore = new SingletoneKCore();
            Map<String, Map<String, Integer>> kcoreNodesMap = new HashMap<>();
            List<String> visited = new ArrayList<>();

            singletoneKCore.renew_KC();  //renewing singletoneKCore (for several files)

            interviews.graphs.Graph InterviewsGraph = new interviews.graphs.Graph(graph.getStemmedTermList_English().size());
            interviews.graphs.KCore InterviewsKCore = new interviews.graphs.KCore(InterviewsGraph);
            graph.graphCreator();
            InterviewsGraph = graph.G1();
            InterviewsKCore = singletoneKCore.KC(graph);
            List<String> stemmedTermList = new ArrayList<>(graph.getStemmedTermList_English());

            for (String s : stemmedTermList) {
                if (!s.equals("NA")) {
                    kcoreNodesMap = new HashMap<>();
                    visited = new ArrayList<>();
                    kcoreNodesList.add(graph.KCoreNodesExtractor(singletoneKCore, s, s, kcoreNodesMap, visited));

                }
            }

            List<Map<String, Map<String, Integer>>> kcoreNodesList1;
            List<Map<String, Map<String, Integer>>> kcoreNodesList2; //contains all kcores that are obtained in a specific step of for loop
            List<Map<String, Map<String, Integer>>> kcoreNodesList_firstTransition = new ArrayList<>();
            LinkedHashMap<Map<String, Map<String, Integer>>, Integer> sortedkcoreNodes_coreNumber_firstTransition = new LinkedHashMap<>();
            Map<String, Map<String, Integer>> allKCoreNodesNLevel; //contains all kcores with a specific (highest) core-number
            while (!graph.getWeightedAdjGraph().isEmpty() || graph.getWeightedAdjGraph().size() != 0) {
                if (isTrue1) {
                    kcoreNodesList = new ArrayList<>();
                    stemmedTermList = new ArrayList<>(graph.getStemmedTermList_English());
                    for (int i = 0; i < stemmedTermList.size(); i++) {
                        if (!stemmedTermList.get(i).equals("NA")) {
                            kcoreNodesMap = new HashMap<>();
                            visited = new ArrayList<>();
                            if (stemmedTermList.get(i).equals("NA")) {
                                int a = 0;
                            }
                            kcoreNodesList.add(graph.KCoreNodesExtractor(singletoneKCore, stemmedTermList.get(i), stemmedTermList.get(i), kcoreNodesMap, visited));
                        }
                    }
                }

                Map<Map<String, Map<String, Integer>>, Integer> kcoreNodes_coreNumber = new HashMap<>();
                int i1 = 0;
                for (Map<String, Map<String, Integer>> m : kcoreNodesList) {

                    try {
                        while (singletoneKCore.KC(graph).core().length > i1 && singletoneKCore.KC(graph).core(i1) == 0) {
                            i1++;
                        }
                        if (singletoneKCore.KC(graph).core().length > i1)
                            kcoreNodes_coreNumber.put(m, singletoneKCore.KC(graph).core(i1));

                        i1++;
                    } catch (Exception e) {
                        e.printStackTrace();
                        System.out.print(m);
                    }
                }

                LinkedHashMap<Map<String, Map<String, Integer>>, Integer> sortedkcoreNodes_coreNumber = new LinkedHashMap<>(graph.sortByValues(kcoreNodes_coreNumber));
                kcoreNodesList1 = new ArrayList<>();
                kcoreNodesList2 = new ArrayList<>();


                for (Map<String, Map<String, Integer>> m : sortedkcoreNodes_coreNumber.keySet()) {
                    kcoreNodesList2.add(m);
                }

                if (!isTrue2) {
                    sortedkcoreNodes_coreNumber_firstTransition = new LinkedHashMap<>(sortedkcoreNodes_coreNumber);
                    isTrue2 = true;
                }
                //computing density of n-leveles of k-cores
                kcoreNodesList2.size();
                int maxCoreNumber = sortedkcoreNodes_coreNumber.get(kcoreNodesList2.get(0));
                allKCoreNodesNLevel = new HashMap<>();
                while (sortedkcoreNodes_coreNumber.get(kcoreNodesList2.get(0)) == maxCoreNumber) {

                    allKCoreNodesNLevel.putAll(kcoreNodesList2.get(0));
                    graph.nodesRemover(kcoreNodesList2.get(0));
                    kcoreNodesList2.remove(kcoreNodesList2.get(0));
                    if (kcoreNodesList2.isEmpty()) {
                        break;
                    }
                }
                kCoreDensity.add(graph.getDensity(allKCoreNodesNLevel));
                isTrue1 = true;
            }
            LinkedHashMap<Integer, Double> level_KcoreDensity = new LinkedHashMap<>();
            for (int i = 0; i < kCoreDensity.size(); i++) {
                level_KcoreDensity.put(i + 1, kCoreDensity.get(i));
            }
            System.out.println(kCoreDensity);
            int elbow = graph.computeElbow(level_KcoreDensity);
            System.out.println("The elbow is:" + elbow);

            //obtaining the KBest-core of main graph as keywords
            List<String> keywordsList = new ArrayList<>();
            int level = 0; //level number in the main graph
            double matchedKeywords = 0; //number of Matched Keywords (for computing precision)
            double totalKeywords = 0; //number of total keywords
            double recoveredKeywords = 0; //number of recovered keywords (for computing recall)
            double totalHumanKeywords = 0; //number of totall human assigned keywords
            boolean isTrue3 = false; //for computing recall.

            List<String> removedkeyWords = new ArrayList<>(); //these keywords should be removed form the corresponding hashmap
            for (Map<String, Map<String, Integer>> map : sortedkcoreNodes_coreNumber_firstTransition.keySet()) {
                //newly added
                if (level + 1 == elbow) {
                    totalKeywords = map.size();
                    for (String s : map.keySet()) {
                        keywordsList.add(s);
                        //System.out.println(s);

                    }

                    //checking accuracy

                    String filePath_test = testFilesPaths.get(fileNumber);
                    //String filePath_test = "/Users/fazeletavakoli/Desktop/KeywordInput_NLPLab/art_and_culture-20918624_t.txt";
                    try {
                        String line = "";
                        BufferedReader br = null;
                        InputStream inputStream = new FileInputStream(filePath_test);
                        br = new BufferedReader(new InputStreamReader(inputStream, "UTF-8"));
                        while ((line = br.readLine()) != null) {
                            isTrue3 = false;
                            totalHumanKeywords++;
                            for (String s : map.keySet()) {
                                if (line.contains(s)) {
                                    matchedKeywords++;
                                    removedkeyWords.add(s);
                                    if (!isTrue3) {
                                        recoveredKeywords++;
                                        isTrue3 = true;
                                    }
                                }
                            }
                            for (String s : removedkeyWords) {
                                map.remove(s);
                            }
                        }
                    } catch (Exception e) {

                    }

                    String commonPath = "/Users/fazeletavakoli/IdeaProjects/stanford-corenlp/testData/";
                    double precision = matchedKeywords / totalKeywords;
                    double recall = recoveredKeywords / totalHumanKeywords;
                    System.out.println("Precision is: " + precision);
                    System.out.println("Recall is: " + recall);
                    String testResult = "Result for " + filePath.substring(commonPath.length()) + ":\t" + "precision: " + precision + "\t Recall: " + recall + "\n";
                    Corpus.writeInFile_version2(testResult, testResultFile);

                    break;
                } else {
                    level++;
                }

            }

        }
    }

}
