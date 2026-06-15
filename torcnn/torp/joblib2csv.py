import sys, argparse
import pandas as pd
import joblib
import pickle 
from os import makedirs, path
from sys import argv
from tqdm import tqdm
from xml import etree

def main(joblibFile, outdir, fileType = "csv"):

   # 1. Read in joblib pipeline safely
   try:
      pipeline = joblib.load(joblibFile)
   except Exception as err:
      print("***ERROR READING JOBLIB FILE***: {}\n{}\nEXITING......".format(joblibFile, err))
      exit()

   # 2. Extract the Random Forest classifier from the pipeline
   # Note: If your pipeline step is named something other than 'rf_classifier', change it here.
   if hasattr(pipeline, 'named_steps'):
      if 'classifier' in pipeline.named_steps:
         rf = pipeline.named_steps['classifier']
      else:
         # Fallback: take the last step of the pipeline if name doesn't match
         rf = pipeline.steps[-1][1]
   else:
      rf = pipeline # If it was just a model and not a pipeline after all

   # 3. Handle Feature Names
   try:
      lookupTable = rf.feature_names_in_ # Updated for modern scikit-learn standard
   except AttributeError:
      try:
         lookupTable = rf.feature_names_
      except AttributeError:
         # Hardcoded severe weather feature lookup table fallback
         lookupTable = pickle.load(open('torp_features_2026Dataset.pkl','rb'))

   if fileType == "xml":
      table = etree.Element("table")
      product = etree.SubElement(table, "product")
      product.set("name", "ProbabilityTornado")
      data = etree.SubElement(table, "data")

      for tree in tqdm(range(rf.n_estimators)):
         # Handle scikit-learn structure difference (.estimators_ instead of treating model as array)
         estimator = rf.estimators_[tree]
         for node in range(estimator.tree_.node_count):
            prob = estimator.tree_.value[node][0][1]/estimator.tree_.weighted_n_node_samples[node]

            item = etree.SubElement(data, "item")
            item.set("tableNumber", "{}".format(tree))
            item.set("node",        "{}".format(node))
            if estimator.tree_.feature[node] == -2:
               item.set("featureName", "leaf")
            else:
               item.set("featureName", "{}".format(lookupTable[estimator.tree_.feature[node]]))
            item.set("leftChild",   "{}".format(estimator.tree_.children_left[node]))
            item.set("rightChild",  "{}".format(estimator.tree_.children_right[node]))
            item.set("threshold",   "{}".format(estimator.tree_.threshold[node]))
            item.set("probability", "{}".format(prob))

      et = etree.ElementTree(table)
      et.write(outdir + "/RandomForest_NTDA.xml", pretty_print=True, xml_declaration=True)

   elif fileType == "csv":
      # Modified list comprehensions below to correctly loop through rf.estimators_[tree] 
      probs   = [rf.estimators_[tree].tree_.value[node][0][1]/rf.estimators_[tree].tree_.weighted_n_node_samples[node] for tree in range(rf.n_estimators) for node in range(rf.estimators_[tree].tree_.node_count)]
      treeNum = [tree for tree in range(rf.n_estimators) for node in range(rf.estimators_[tree].tree_.node_count)]
      nodeNum = [node for tree in range(rf.n_estimators) for node in range(rf.estimators_[tree].tree_.node_count)]
      feature = ["leaf" if rf.estimators_[tree].tree_.feature[node] == -2 else lookupTable[rf.estimators_[tree].tree_.feature[node]] for tree in range(rf.n_estimators) for node in range(rf.estimators_[tree].tree_.node_count)]
      left    = [rf.estimators_[tree].tree_.children_left[node] for tree in range(rf.n_estimators) for node in range(rf.estimators_[tree].tree_.node_count)]
      right   = [rf.estimators_[tree].tree_.children_right[node] for tree in range(rf.n_estimators) for node in range(rf.estimators_[tree].tree_.node_count)]
      thresh  = [rf.estimators_[tree].tree_.threshold[node] for tree in range(rf.n_estimators) for node in range(rf.estimators_[tree].tree_.node_count)]

      df = pd.DataFrame({'tableNumber':treeNum, 'node':nodeNum, 'featureName':feature, 'leftChild':left, 'rightChild':right, 'threshold':thresh, 'probability':probs})
      df.to_csv(path_or_buf = str('{}/RandomForest_NTDA.csv'.format(outdir)), index=False)

   return 0

if __name__ == "__main__":
   parser = argparse.ArgumentParser(description='Reads in random forest joblib file, and makes an XML or CSV file.')
   parser.add_argument('-f', metavar='fileType', type=str, nargs='?', default="csv", help='Specify output file type.')
   parser.add_argument('joblibFile', type=str, nargs=1, help='Path to random forest joblib file')
   parser.add_argument('outputDir',  type=str, nargs=1, help='Output directory where the output file will be saved')
   args = parser.parse_args(argv[1:])

   if not path.exists(args.joblibFile[0]):
      print("{} does not exist. Please restart with valid data.".format(args.joblibFile[0]))
      exit()

   if not path.exists(args.outputDir[0]):
      create_dir = input("The directory {} does not exist. Would you like to create it? (y/n): ".format(args.outputDir[0])).strip().lower()

      if create_dir == 'y':
         try:
            makedirs(args.outputDir[0])
            print("Directory {} created successfully.".format(args.outputDir[0]))
         except Exception as e:
            print("An error occurred while creating the directory: {}".format(e))
            exit()
      else:
         print("Directory creation was canceled. Restart with valid output directory. Exiting.")
         exit()

   main(args.joblibFile[0], args.outputDir[0], fileType = args.f)
