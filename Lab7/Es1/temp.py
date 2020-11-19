import pandas as pd
import numpy as np

file_ids = ['174.wav', '354.wav', '344.wav', '233.wav', '168.wav', '483.wav', '459.wav', '46.wav', '100.wav', '348.wav',
            '154.wav', '99.wav', '123.wav', '80.wav', '427.wav', '451.wav', '298.wav', '27.wav', '231.wav', '112.wav',
            '381.wav', '224.wav', '497.wav', '0.wav', '361.wav', '155.wav', '367.wav', '377.wav', '312.wav', '357.wav',
            '273.wav', '356.wav', '349.wav', '120.wav', '147.wav', '209.wav', '131.wav', '228.wav', '238.wav',
            '327.wav',
            '328.wav', '119.wav', '163.wav', '79.wav', '355.wav', '21.wav', '92.wav', '341.wav', '115.wav', '347.wav',
            '365.wav', '176.wav', '490.wav', '90.wav', '188.wav', '477.wav', '253.wav', '424.wav', '308.wav', '68.wav',
            '283.wav', '372.wav', '385.wav', '429.wav', '87.wav', '470.wav', '258.wav', '441.wav', '59.wav', '25.wav',
            '329.wav', '203.wav', '321.wav', '139.wav', '392.wav', '183.wav', '352.wav', '178.wav', '181.wav',
            '293.wav',
            '240.wav', '91.wav', '301.wav', '218.wav', '36.wav', '447.wav', '58.wav', '364.wav', '339.wav', '266.wav',
            '61.wav', '217.wav', '386.wav', '196.wav', '473.wav', '452.wav', '288.wav', '158.wav', '468.wav', '439.wav',
            '430.wav', '280.wav', '44.wav', '485.wav', '172.wav', '47.wav', '443.wav', '226.wav', '422.wav', '338.wav',
            '48.wav', '395.wav', '291.wav', '249.wav', '267.wav', '153.wav', '469.wav', '45.wav', '378.wav', '460.wav',
            '210.wav', '132.wav', '2.wav', '409.wav', '111.wav', '175.wav', '488.wav', '236.wav', '157.wav', '247.wav',
            '4.wav', '124.wav', '167.wav', '492.wav', '432.wav', '462.wav', '84.wav', '38.wav', '23.wav', '335.wav',
            '107.wav',
            '33.wav', '330.wav', '22.wav', '127.wav', '486.wav', '491.wav', '311.wav', '489.wav', '104.wav', '207.wav',
            '404.wav', '235.wav', '407.wav', '417.wav', '96.wav', '108.wav', '55.wav', '284.wav', '285.wav', '49.wav',
            '276.wav', '15.wav', '496.wav', '297.wav', '346.wav', '93.wav', '234.wav', '1.wav', '315.wav', '388.wav',
            '261.wav', '498.wav', '134.wav', '110.wav', '28.wav', '360.wav', '199.wav', '426.wav', '475.wav', '484.wav',
            '72.wav', '300.wav', '465.wav', '390.wav', '67.wav', '215.wav', '248.wav', '370.wav', '54.wav', '406.wav',
            '213.wav', '114.wav', '353.wav', '487.wav', '331.wav', '37.wav', '292.wav', '164.wav', '369.wav', '219.wav',
            '206.wav', '186.wav', '271.wav', '76.wav', '40.wav', '39.wav', '77.wav', '334.wav', '89.wav', '320.wav',
            '161.wav',
            '379.wav', '11.wav', '442.wav', '149.wav', '184.wav', '64.wav', '230.wav', '421.wav', '29.wav', '337.wav',
            '318.wav', '394.wav', '190.wav', '431.wav', '461.wav', '456.wav', '402.wav', '137.wav', '384.wav',
            '289.wav',
            '414.wav', '499.wav', '35.wav', '116.wav', '433.wav', '399.wav', '317.wav', '212.wav', '31.wav', '229.wav',
            '380.wav', '374.wav', '223.wav', '86.wav', '455.wav', '373.wav', '66.wav', '270.wav', '171.wav', '476.wav',
            '73.wav', '382.wav', '396.wav', '126.wav', '403.wav', '294.wav', '98.wav', '326.wav', '237.wav', '336.wav',
            '187.wav', '129.wav', '202.wav', '241.wav', '256.wav', '121.wav', '398.wav', '375.wav', '200.wav',
            '444.wav',
            '411.wav', '363.wav', '32.wav', '453.wav', '458.wav', '198.wav', '152.wav', '454.wav', '16.wav', '383.wav',
            '140.wav', '130.wav', '222.wav', '52.wav', '156.wav', '397.wav', '83.wav', '410.wav', '286.wav', '250.wav',
            '3.wav', '197.wav', '345.wav', '78.wav', '141.wav', '495.wav', '448.wav', '109.wav', '95.wav', '401.wav',
            '102.wav', '268.wav', '118.wav', '243.wav', '180.wav', '457.wav', '101.wav', '305.wav', '425.wav',
            '287.wav',
            '144.wav', '282.wav', '160.wav', '193.wav', '204.wav', '69.wav', '314.wav', '265.wav', '303.wav', '272.wav',
            '393.wav', '464.wav', '299.wav', '135.wav', '434.wav', '143.wav', '259.wav', '436.wav', '428.wav',
            '170.wav',
            '9.wav', '20.wav', '146.wav', '368.wav', '128.wav', '482.wav', '343.wav', '478.wav', '419.wav', '166.wav',
            '340.wav', '387.wav', '75.wav', '389.wav', '51.wav', '191.wav', '65.wav', '319.wav', '467.wav', '136.wav',
            '26.wav', '5.wav', '281.wav', '359.wav', '493.wav', '53.wav', '332.wav', '472.wav', '302.wav', '34.wav',
            '12.wav',
            '440.wav', '466.wav', '316.wav', '313.wav', '304.wav', '290.wav', '185.wav', '362.wav', '192.wav', '97.wav',
            '262.wav', '81.wav', '246.wav', '405.wav', '43.wav', '63.wav', '85.wav', '358.wav', '195.wav', '225.wav',
            '62.wav',
            '413.wav', '138.wav', '211.wav', '408.wav', '471.wav', '415.wav', '71.wav', '201.wav', '56.wav', '88.wav',
            '142.wav', '494.wav', '366.wav', '19.wav', '194.wav', '412.wav', '423.wav', '94.wav', '296.wav', '418.wav',
            '333.wav', '179.wav', '70.wav', '351.wav', '479.wav', '480.wav', '227.wav', '438.wav', '322.wav', '450.wav',
            '18.wav', '103.wav', '10.wav', '205.wav', '400.wav', '279.wav', '446.wav', '481.wav', '435.wav', '306.wav',
            '244.wav', '325.wav', '274.wav', '60.wav', '151.wav', '232.wav', '50.wav', '245.wav', '254.wav', '437.wav',
            '57.wav', '371.wav', '277.wav', '125.wav', '17.wav', '74.wav', '24.wav', '264.wav', '445.wav', '278.wav',
            '307.wav', '7.wav', '310.wav', '162.wav', '189.wav', '13.wav', '309.wav', '177.wav', '324.wav', '173.wav',
            '449.wav', '133.wav', '275.wav', '30.wav', '216.wav', '251.wav', '165.wav', '41.wav', '474.wav', '260.wav',
            '42.wav', '105.wav', '391.wav', '14.wav', '159.wav', '269.wav', '106.wav', '416.wav', '8.wav', '350.wav',
            '220.wav', '252.wav', '150.wav', '113.wav', '239.wav', '208.wav', '342.wav', '257.wav', '242.wav',
            '295.wav',
            '169.wav', '323.wav', '376.wav', '117.wav', '182.wav', '263.wav', '122.wav', '6.wav', '463.wav', '420.wav',
            '148.wav', '145.wav', '82.wav', '221.wav', '214.wav', '255.wav']
labels = ['2', '1', '6', '3', '1', '4', '8', '9', '0', '2', '7', '4', '6', '3', '1', '9', '4', '4',
          '5', '2', '8', '4', '1',
          '6', '5', '6', '1', '0', '0', '4', '3', '7', '6', '4', '6', '5'
    , '2', '3', '9', '1', '3', '6', '8', '8', '7', '7', '4', '3', '1', '3', '8', '2', '9', '4'
    , '4', '7', '9', '9', '0', '1', '2', '3', '5', '6', '4', '6', '7', '5', '2', '5', '4', '3'
    , '2', '3', '6', '7', '0', '8', '0', '8', '8', '2', '5', '7', '4', '8', '7', '6', '3', '2'
    , '7', '1', '9', '9', '6', '7', '4', '0', '0', '6', '8', '4', '6', '9', '4', '7', '4', '1'
    , '6', '3', '0', '5', '2', '5', '5', '6', '4', '3', '2', '2', '2', '3', '3', '1', '1', '8'
    , '1', '4', '7', '1', '6', '4', '0', '5', '3', '9', '1', '5', '1', '3', '1', '8', '1', '4'
    , '0', '0', '6', '8', '3', '2', '2', '2', '5', '1', '4', '7', '2', '5', '8', '1', '7', '1'
    , '8', '6', '1', '0', '6', '3', '9', '5', '8', '3', '7', '0', '3', '1', '4', '9', '6', '0'
    , '6', '9', '5', '9', '3', '1', '3', '2', '5', '1', '7', '2', '8', '8', '9', '7', '0', '7'
    , '9', '4', '7', '2', '8', '0', '8', '0', '1', '4', '4', '3', '5', '4', '7', '4', '2', '1'
    , '4', '5', '6', '8', '9', '8', '8', '5', '7', '2', '3', '4', '9', '0', '0', '2', '0', '8'
    , '0', '5', '6', '1', '0', '8', '1', '6', '8', '4', '5', '3', '2', '2', '9', '7', '1', '9'
    , '2', '0', '5', '8', '4', '2', '1', '3', '4', '0', '4', '8', '4', '2', '5', '0', '9', '4'
    , '6', '5', '4', '1', '1', '4', '8', '9', '3', '1', '4', '5', '2', '7', '4', '6', '1', '0'
    , '8', '4', '2', '4', '1', '7', '9', '6', '8', '9', '0', '1', '2', '4', '7', '6', '4', '9'
    , '9', '9', '4', '9', '9', '3', '3', '0', '5', '4', '5', '2', '6', '4', '0', '6', '8', '2'
    , '8', '8', '6', '4', '8', '3', '4', '2', '5', '8', '9', '4', '1', '2', '5', '5', '1', '2'
    , '6', '4', '9', '9', '0', '9', '1', '2', '6', '3', '6', '5', '1', '3', '9', '0', '8', '9'
    , '4', '9', '5', '5', '8', '2', '1', '0', '0', '6', '6', '6', '4', '4', '5', '6', '1', '3'
    , '6', '7', '4', '6', '4', '3', '3', '1', '3', '1', '1', '4', '2', '5', '9', '1', '1', '1'
    , '5', '0', '6', '3', '2', '0', '2', '6', '5', '5', '2', '2', '6', '2', '8', '8', '6', '4'
    , '4', '5', '2', '4', '5', '3', '9', '3', '7', '8', '0', '4', '4', '3', '6', '3', '2', '9'
    , '9', '0', '8', '1', '5', '3', '6', '4', '2', '7', '2', '5', '8', '4', '3', '8', '2', '0'
    , '2', '2', '2', '0', '0', '5', '2', '6', '8', '9', '2', '2', '0', '5', '5', '2', '4', '8'
    , '5', '7', '8', '5', '0', '0', '3', '9', '9', '1', '6', '7', '2', '4', '3', '0', '1', '2'
    , '0', '9', '0', '0', '8', '8', '4', '9', '8', '3', '5', '7', '6', '1']

df = pd.DataFrame(data=np.zeros(500), columns=["Predicted"])

with open("myfile.csv", "w") as f:
    f.write("Id,Predicted\n")
    for file_id, label in zip(file_ids, labels):
        riga = int(file_id.split('.')[0])
        df.iloc[riga, [0]] = label

    print(df)

    for i in range(df.shape[0]):
        print(df.iloc[i, 0])
        stringa = str(i) + "," + str(df.iloc[i, 0] + "\n")
        f.write(stringa)
