{\rtf1\ansi\ansicpg1252\cocoartf2822
\cocoatextscaling0\cocoaplatform0{\fonttbl\f0\fmodern\fcharset0 Courier;}
{\colortbl;\red255\green255\blue255;\red131\green0\blue165;\red255\green255\blue255;\red0\green0\blue0;
\red144\green1\blue18;\red15\green112\blue1;\red0\green0\blue255;\red31\green99\blue128;\red19\green85\blue52;
}
{\*\expandedcolortbl;;\cssrgb\c59216\c13725\c70588;\cssrgb\c100000\c100000\c100000;\cssrgb\c0\c0\c0;
\cssrgb\c63922\c8235\c8235;\cssrgb\c0\c50196\c0;\cssrgb\c0\c0\c100000;\cssrgb\c14510\c46275\c57647;\cssrgb\c6667\c40000\c26667;
}
\paperw11900\paperh16840\margl1440\margr1440\vieww25400\viewh15980\viewkind0
\deftab720
\pard\pardeftab720\partightenfactor0

\f0\fs28 \cf2 \cb3 \expnd0\expndtw0\kerning0
\outl0\strokewidth0 \strokec2 import\cf0 \strokec4  torch\cb1 \
\cf2 \cb3 \strokec2 from\cf0 \strokec4  torch.utils.data \cf2 \strokec2 import\cf0 \strokec4  DataLoader, random_split\cb1 \
\cf2 \cb3 \strokec2 import\cf0 \strokec4  numpy \cf2 \strokec2 as\cf0 \strokec4  np\cb1 \
\cf2 \cb3 \strokec2 import\cf0 \strokec4  os\cb1 \
\
\
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 time_series_files = os.listdir(\cf5 \strokec5 "/content/drive/MyDrive/Github_projects/Double_pendulum/dataset/timeseries"\cf0 \strokec4 )\cb1 \
\
\pard\pardeftab720\partightenfactor0
\cf6 \cb3 \strokec6 # Load all 'coords' tensors into a list\cf0 \cb1 \strokec4 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3 coords_tensors = []\cb1 \
\pard\pardeftab720\partightenfactor0
\cf2 \cb3 \strokec2 for\cf0 \strokec4  file_name \cf7 \strokec7 in\cf0 \strokec4  time_series_files:\cb1 \
\pard\pardeftab720\partightenfactor0
\cf0 \cb3     file_path = os.path.join(\cf5 \strokec5 "/content/drive/MyDrive/Github_projects/Double_pendulum/dataset/timeseries"\cf0 \strokec4 , file_name)\cb1 \
\cb3     \cf6 \strokec6 # Load the 'coords' array and convert it to a PyTorch tensor\cf0 \cb1 \strokec4 \
\cb3     tensor = torch.from_numpy(np.load(file_path)[\cf5 \strokec5 'coords'\cf0 \strokec4 ])\cb1 \
\cb3     coords_tensors.append(tensor)\cb1 \
\cb3 coords_tensors = torch.stack(coords_tensors)\cb1 \
\
\
\cb3 number_of_validation_data, number_of_testing_data = \cf8 \cb3 \strokec8 int\cf0 \cb3 \strokec4 (\cf9 \cb3 \strokec9 0.1\cf0 \cb3 \strokec4  * coords_tensors.shape[\cf9 \cb3 \strokec9 0\cf0 \cb3 \strokec4 ]), \cf8 \cb3 \strokec8 int\cf0 \cb3 \strokec4 (\cf9 \cb3 \strokec9 0.1\cf0 \cb3 \strokec4  * coords_tensors.shape[\cf9 \cb3 \strokec9 0\cf0 \cb3 \strokec4 ])\cb1 \
\cb3 number_of_training_data = \cf8 \cb3 \strokec8 int\cf0 \cb3 \strokec4 (coords_tensors.shape[\cf9 \cb3 \strokec9 0\cf0 \cb3 \strokec4 ] - number_of_validation_data - number_of_testing_data)\cb1 \
\
\cb3 testing_dataset, validation_dataset, training_dataset = random_split(coords_tensors, [number_of_testing_data, number_of_validation_data, number_of_training_data],torch.Generator().manual_seed(\cf9 \cb3 \strokec9 42\cf0 \cb3 \strokec4 ))\cb1 \
}