/* global ApexCharts */

function generateChart(containerId, yValid, yHat) {
  const chartContainer = document.querySelector(containerId);

  if (!chartContainer) {
    return;
  }

  var options = {
    series: [
      {
        name: "y_valid",
        data: yValid,
      },
      {
        name: "yhat",
        data: yHat,
      },
    ],
    colors: ["#FFFFFF", "#EE771B"],
    chart: {
      type: "line",
      height: 400,
      toolbar: { show: true },
      zoom: { enabled: true },
    },
    stroke: {
      curve: "smooth",
    },
    xaxis: {
      labels: { show: false },
      axisTicks: { show: false },
      axisBorder: { show: false },
    },
    yaxis: {
      labels: { show: false },
      axisTicks: { show: false },
      axisBorder: { show: false },
    },
    tooltip: { enabled: false },
    grid: { show: false },
    legend: { show: false },
  };

  var chart = new ApexCharts(document.querySelector(containerId), options);
  chart.render();
}

document.addEventListener("DOMContentLoaded", function () {
  generateChart(
    "#chart1",
    [
      0.5956779614659786, -0.04333705995088488, -0.6881603028193747,
      0.06022675139269685, -0.6254347368239372, -0.37148999777670194,
      0.6105873387542123, 0.6887954978484564, 0.7837361024864593,
      -0.5241085851785254, 0.20128513456776725, -0.8253180170588841,
      0.47082645174654314, -0.6833628821066886, -0.3617406848948889,
      0.6856485511382346, -0.15337582483097287, -0.1618682191594043,
      0.5717198792018168, -0.050572927744164324, -0.35723649215618275,
      -0.19334240304710287, 0.15479906525939321, -0.5482499716169706,
      -0.7379563748897571, -0.9925494330460152, -0.6044912134027072,
      0.3185855613079674, 0.7686115105589897, -0.15932667874548867,
      0.6378654496132169, 0.6060824433170097, 0.0680765580271367,
      -0.3021427528659532, -0.545321765125144, -0.6732424871437689,
      -0.8985477735493932, -0.2728298067007159, 0.30760733341609275,
      0.4340776549439967, 0.42495112341221924, -0.7100288797867481,
      -0.2668222714895989, -0.6688635799230885, 0.008529433679739242,
      0.8714735350015809, -0.07035263876953331, 0.1602895251892767,
      0.5584788674836019, 1.0042908366141459,
    ],
    [
      0.5956779614659786, -0.04333705995088488, -0.688125509896666,
      0.060186712552984904, -0.6253545413685471, -0.37136753740629597,
      0.6105654690356151, 0.688791075351491, 0.7836317825288164,
      -0.5243070305631288, 0.20122424683684448, -0.8252006703862675,
      0.47084699336267133, -0.6833622080919415, -0.3616292679102178,
      0.6854930578549306, -0.15341284816664075, -0.1617323160208413,
      0.5716641225120068, -0.050537413899616246, -0.35729490178138296,
      -0.1934069285111186, 0.15472996858645047, -0.5482262208432439,
      -0.7379266350672994, -0.9924983637821746, -0.6044424014392532,
      0.3187865824775367, 0.7685600313256096, -0.15949959395883267,
      0.6378077322573125, 0.6058972417826072, 0.06811599955979122,
      -0.3022914620327069, -0.545456611000648, -0.6731533782960734,
      -0.898424358391404, -0.27274397681166185, 0.3076036495459946,
      0.43406905399695844, 0.4248616774131386, -0.7100028470709036,
      -0.2669592847713398, -0.6689915887883622, 0.008481626257395694,
      0.8713958038340013, -0.07041072802578202, 0.16017671955956642,
      0.5583870182693625, 1.0041427109466916,
    ],
  );
  generateChart(
    "#chart2",
    [
      -0.15154484921168657, 0.12372369616072168, 0.5230985187868696,
      -0.3896166974455501, 0.6058506686815857, -0.46440177179021097,
      0.05568931092900589, 0.824524205800903, 0.5665201633115969,
      -0.6178263945438314, -0.6659972257962191, 0.5690574365118961,
      0.34025508531287285, 0.14217141525704546, -0.20285316328590486,
      -0.44124445695647124, -0.1506196918352653, -0.8736656970675117,
      -0.29543576888746875, -0.25836942878356, -0.8857270203414978,
      -0.5081356048509796, 0.07578984216116866, -0.07099586145055194,
      -0.7352726667516855, -0.0580329605198542, -0.6627261799479609,
      -0.8442115745005042, -0.08245860477073981, 0.1632747121706431,
      0.6648139706343034, -0.43698043168640105, -0.6362192483725886,
      0.5027523333505042, -0.3824970789903021, -0.7921334482339535,
      -0.4564097948963876, -0.26209209591050636, 0.7551996850201524,
      0.8259556645612789, -0.0008935572781436175, -0.7481368147071129,
      0.6727752465710547, -0.10341496544698249, -0.22841448889436874,
      -0.5723782996323984, 0.5577888903560067, -0.6126634925514718,
      -0.5825817031735219, 0.38559553610647007,
    ],
    [
      -0.15154484921168657, 0.12372369616072168, 0.5280475000414957,
      -0.4277543628085869, 0.5688679963224945, -0.486639731459004,
      0.099149353400382, 0.8194973867890675, 0.6247360084043121,
      -0.5797078845505547, -0.6038171229458594, 0.5936307359793414,
      0.34047822291794455, 0.1538318172787278, -0.1901124551485444,
      -0.4366835304331323, -0.18381839094456498, -0.886762814518636,
      -0.3106761747572755, -0.2824465996353984, 0.9031919816251405,
      -0.4859547250484367, 0.08193867029234639, -0.058993088094603,
      -0.7343038908108176, -0.10308453150712835, -0.6763706597488855,
      -0.83755115806315, -0.05508494319193667, 0.17833141073337733,
      0.671569862393888, -0.3988069909128267, -0.5957348902322156,
      0.47881605471046523, -0.34587161957545703, -0.7984239936788731,
      -0.4606676168833024, -0.21369325027970582, 0.7751182325435183,
      0.8282641467250256, 0.06921640756104384, -0.7370936091997332,
      0.6500926919999714, -0.0902601220277811, -0.2263157899288335,
      -0.5528212180045154, 0.5154920305634874, -0.5989925601816369,
      -0.5365904484952801, 0.4355297014571458,
    ],
  );
  generateChart(
    "#chart3",
    [
      -0.49801223345117385, -0.17047109462530446, -0.6279580488979899,
      0.49263380851256156, -0.2854986517807117, -0.5635327888100402,
      -8.789864406474828e-5, 0.557705748715272, -0.3665724726307606,
      0.04395535888000136, 0.7203328427859333, 1.0107576404796592,
      0.3067698443780213, 0.3788806821268591, 0.5499886248904351,
      0.8040131743746624, -0.6327054027715229, 0.1678025764778953,
      0.707072218955234, -0.6081389172322609, -0.5539811161040263,
      0.7114300546572634, -0.5337399017081289, -0.009716947939309805,
      -0.6036812266778026, 0.7968893387360255, -0.2756442349389169,
      -0.31241518191578643, 0.6476820772953411, -0.6277428142512554,
      0.19428680295486903, -0.26323868053006516, 0.3876953965557445,
      -0.38220583449843415, 0.5032714065775519, 0.12936779273886562,
      -0.08934327216706393, 0.2838803387308956, -0.8252978689644223,
      -0.6192236943315822, -0.5914839932268481, 0.4507783452219252,
      0.3694073410638243, 0.08347434526190108, 0.21668584532568094,
      0.2887045197596661, 0.8048649331695101, -0.4752582071892234,
      0.13594875546450977, 0.45051956408164134,
    ],
    [
      -0.49801223345117385, -0.17047109462530446, -0.628071638685502,
      0.4926106924920685, -0.2855678596287499, -0.5635339130855337,
      4.3926194226627524e-5, 0.5577105984236594, -0.36662567330596335,
      0.04404574470061495, 0.7204914870177367, 1.0107616837680944,
      0.3069154942715836, 0.37906497345019563, 0.5500549136688777,
      0.8039704174572794, -0.6325643053952457, 0.16795925897526987,
      0.7069504742323925, -0.6082051826860608, -0.5539162095066537,
      0.7113895892915333, -0.533666910850649, -0.009731442878754402,
      -0.6035456705317491, 0.7968028094468553, -0.2757328200091035,
      -0.31235344716458435, 0.6477398644943517, -0.6278167713640124,
      0.19420813744074233, -0.2633303159457468, 0.3876526987060168,
      -0.3821845681773225, 0.5033194894742398, 0.12941541331794576,
      -0.08924371049954051, 0.2837259400805959, -0.8253644247449702,
      -0.6190883740269181, -0.5918234051104178, 0.450670746693622,
      0.36939358672508027, 0.08356988186273295, 0.21680826319728458,
      0.2887440643156897, 0.8047901152405835, -0.47530604381978625,
      0.1360703799896089, 0.4504487462874382,
    ],
  );
  generateChart(
    "#chart4",
    [
      0.7196231067608067, -0.45524167145271016, -0.007278924931139515,
      0.5852046433498619, -0.4631852629096133, -0.25331825077072223,
      -0.6118266499500274, -0.29008874231868975, 0.46623658654527617,
      0.8860065239516365, 0.8760416160789268, 0.2256735055353557,
      0.511339393463504, -0.32507142936162436, 0.21195973683334102,
      0.9295751901875017, -0.0964822339179365, -0.25103766744651324,
      0.5326333018443066, 0.01619871566194143, 0.013014170494765142,
      0.3363176613586215, -0.8029425810856741, -0.6689381053679783,
      0.7077708995451378, -0.010865286156132862, 0.22571916111602291,
      -0.6952057405180423, -0.39449856552303814, -0.5553034547275404,
      -0.9119039808070999, 0.2099167462083274, -0.25902626736504375,
      0.4483141064154227, 0.03642481057867556, -0.432685582034749,
      -0.1021473576336465, -0.7530552914616662, -0.4727322784500165,
      0.6484035826083212, -0.5862509385934896, -0.7954488496483639,
      -0.022350749571675085, 0.38721429804510515, 0.7999943244268793,
      -0.49090763107179375, -0.08392155587890898, 0.8089504284259561,
      0.34909012836252784, 0.8982623317354252,
    ],
    [
      0.7196231, -0.45524168, 0.0059637725, 0.5795059, -0.4651241, -0.23806365,
      -0.60254955, -0.27806705, 0.49408773, 0.8838929, 0.87453043, 0.23316672,
      0.5146867, -0.3231173, 0.20901921, 0.9142345, -0.088802494, -0.2643619,
      0.5368148, 0.037154436, 0.022096843, 0.3388984, -0.79941666, -0.64389503,
      0.70165896, -0.03400141, 0.22868821, -0.6917361, -0.38799596, -0.55117804,
      -0.88060963, 0.22225848, -0.23744297, 0.46197364, 0.056968212, -0.4466999,
      -0.10601547, -0.73626244, -0.4704472, 0.6488956, -0.58015406, -0.793458,
      -0.005146891, 0.39878616, 0.7844018, -0.49513453, -0.07138334, 0.8003892,
      0.34460956, 0.9006995,
    ],
  );
  generateChart(
    "#chart5",
    [
      0.14409487714702274, -0.7451149014676122, 0.18793431090477744,
      0.6886446501808147, 0.6714032980980876, 0.4529678011774795,
      -0.07259514587980916, 0.7129356752880055, -0.09204591119633194,
      0.5827365732214531, 0.5387136612117085, 0.7654831525695498,
      0.8495241810161596, 0.9298098336968017, 0.1750057104470646,
      0.3074749283352481, -0.24298386492194957, -0.6939806619559009,
      -0.5687658853167273, -0.5890705192408073, 0.8170728904426371,
      -0.450318475528044, -0.9523565437309297, -0.4365858302297869,
      -0.6840228939548312, 0.6312516129242433, 1.030516948053811,
      0.4251196786189155, -0.09797656654293013, -0.6309878699939107,
      -0.05573123465396932, -0.4905591698249629, 0.756635189591348,
      -0.588465244112538, 0.3792057158182954, -0.008426684298423363,
      0.5679577615543134, 0.9606851506364256, 0.6832366926892103,
      0.6197148786830832, 0.4735872242668437, 0.0629802804466102,
      0.6711719023416661, -0.12514892492082555, -0.6510255413933349,
      0.7536300590268005, -0.184084427237483, 0.7601663093393484,
      -0.5596265620088792, 0.40684707289161165,
    ],
    [
      0.14409487714702274, -0.7451149014676122, 0.18787267020020623,
      0.6886088357031713, 0.6713545644391126, 0.4529230783841014,
      -0.07261388499947885, 0.7130354691756544, -0.09213448975753633,
      0.5828311228927748, 0.538670705497812, 0.7656102557071751,
      0.8496091486479568, 0.9297446878649065, 0.1750730889135143,
      0.3075324739953631, -0.24283728853461012, -0.6940032680789976,
      -0.5687768276686579, -0.5891275801145266, 0.8169647220349742,
      -0.4503732841605199, -0.9522392915213735, -0.4365206601246022,
      -0.6842258063622142, 0.6311852461400167, 1.0305276864896287,
      0.42529320699560713, -0.09790057659262395, -0.6309358626369341,
      -0.055446316642052317, -0.49047712412432587, 0.7567689599089065,
      -0.5885474663674677, 0.3789809755659354, -0.008647719421641921,
      0.5677972868151143, 0.9604663955444398, 0.6830236903624434,
      0.6196313043147192, 0.473666111403501, 0.06303108270141629,
      0.6713660367868272, -0.12498051166360147, -0.6510241267710559,
      0.75369803610687, -0.183931671727898, 0.7603389042093791,
      -0.5593743692362904, 0.4068531135010112,
    ],
  );
  generateChart(
    "#chart6",
    [
      0.4598573971619478, -0.05449346693263307, 0.3618766799667739,
      -0.6393830324582649, -0.46295053601254055, -0.8145604147990776,
      -0.5600821591055863, -0.8119117382092319, -0.8715919459031438,
      -0.11240666180875547, 0.0962141589805026, 0.5545028761000689,
      0.03825307929279026, 0.24302566730818462, 0.41626751634398584,
      0.4545002193311268, 0.7846256551169299, -0.6870589923904427,
      -0.30589611451985843, -0.08014325469924397, -0.07931571482634936,
      0.8728515414294112, -0.3494599969230043, 0.1979082072328673,
      -0.08806445010041633, 0.7525800885864352, 0.15229219644760048,
      -0.19389286146355958, 0.19479067938188865, -0.19427694048020852,
      -0.17553328088229786, -0.678578183719451, -0.5407891567581196,
      -0.5469432414978889, -0.05323492277170957, 0.4496445133539449,
      0.9151995358890433, -0.2391647354790714, -0.4330195550028603,
      -0.1992545229377998, -0.35202355836262345, 0.26100757671655955,
      0.7955227062856352, 0.4715350145554475, 0.1745434843938684,
      0.3086164284715996, 0.6226895927318984, 0.7812852203311796,
      -0.24808081867462942, -0.3093530328212763,
    ],
    [
      0.4598573971619478, -0.05449346693263307, 0.38387930717756324,
      -0.6450095767320296, -0.46173245891383236, -0.7993455475387045,
      -0.5311051541123728, -0.8120946688175402, -0.8633133111816899,
      -0.08619469320619241, 0.12180636116923128, 0.5492243608133278,
      0.03260657665431163, 0.22548372822419713, 0.4054993928856367,
      0.46309456043083913, 0.7895621229558571, -0.7069795166056156,
      -0.2762232002731852, -0.03550317690604157, -0.06811909434821033,
      0.8535888141566786, -0.3669422120824232, 0.22498555898462605,
      -0.08713321710660031, 0.7026792442056803, 0.12204414320698853,
      -0.19866094603488077, 0.17436649623728054, -0.18929070772213738,
      -0.18691902025801868, -0.7151577660128341, -0.5411583057400954,
      -0.5576868269433525, -0.06794735005525124, 0.44778469538198235,
      0.9168727325292902, -0.2515672490373163, -0.44390530182189786,
      -0.20451930714735642, -0.357861791309607, 0.2931779875696077,
      0.8191275449839474, 0.4725087603136875, 0.18080735638389153,
      0.3074828664953468, 0.6155151838159927, 0.7840626281909767,
      -0.26141260638089364, -0.30589835199685467,
    ],
  );
});
