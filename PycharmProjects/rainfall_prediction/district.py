di_arr = ['KANKER', 'SALEM', 'KAWARDHA', 'RAE BARELI', 'KOLHAPUR', 'BID', 'SHIVPURI', 'GAYA', 'GARHWA', 'DAVANAGERE',
          'FIROZABAD', 'HINGOLI', 'CHITRAKOOT', 'NELLORE', 'BAGHPAT', 'GOPALGANJ', 'SOUTH TWENTY FOUR PARGANA', 'MOGA',
          'RAYAGADA', 'BULANDSHAHAR', 'FATEHPUR', 'CHIKMAGALUR', 'MAHBUBNAGAR', 'KHORDHA', 'OSMANABAD', 'CHAMPAWAT',
          'GAJAPATI', 'DHAR', 'KAIMUR (BHABUA)', 'SAMASTIPUR', 'JALGAON', 'DHAMTARI', 'AMRELI', 'HAILAKANDI',
          'SRINAGAR','DHARWAD', 'BARDDHAMAN', 'EAST KHASI HILLS', 'SULTANPUR', 'BHAVNAGAR', 'SOLAN', 'BARAMULA',
          'UMARIA', 'GORAKHPUR', 'TUMKUR', 'KORAPUT', 'EAST GODAVARI', 'WARDHA', 'NORTH SIKKIM', 'KALAHANDI', 'ALIGARH',
          'VAISHALI', 'TIRUCHIRAPALLI', 'SUPAUL', 'ERODE', 'NORTH CACHAR HILLS', 'TAWANG', 'FIROZPUR', 'SAHARSA',
          'KINNAUR', 'PILIBHIT', 'MANDI', 'KOHIMA', 'PATNA', 'HISAR', 'KOCH BIHAR', 'AMRAVATI', 'BULDANA',
          'NAGAPATTINAM', 'KATIHAR', 'BHARATPUR', 'MURSHIDABAD', 'MAHOBA', 'AZAMGARH', 'BALANGIR', 'CHURACHANDPUR',
          'BHOPAL', 'MAYURBHANJ', 'SANT KABIR NAGAR', 'REWARI', 'EAST GARO HILLS', 'SOLAPUR', 'BAHRAICH', 'SURAT',
          'HYDERABAD', 'FARRUKHABAD', 'SIBSAGAR', 'SEONI', 'WEST NIMAR', 'IMPHAL EAST', 'BHOJPUR', 'KARGIL',
          'MADHUBANI', 'PURBI SINGHBHUM', 'BARMER', 'ZUNHEBOTO', 'UNNAO', 'DHANBAD', 'KARAULI', 'NAGAUR', 'BISHNUPUR',
          'JALPAIGURI', 'KURNOOL', 'MEDAK', 'ALMORA', 'BELLARY', 'KAITHAL', 'RAISEN', 'SENAPATI', 'CUDDALORE', 'PALAMU',
          'AMRITSAR', 'KOTA', 'ARIYALUR', 'HAZARIBAGH', 'AHMADNAGAR', 'TINSUKIA', 'SHRAWASTI', 'PURULIYA', 'DEWAS',
          'DOHAD', 'LOHARDAGA', 'KARUR', 'SHIMLA', 'PUDUKKOTTAI', 'KANDHAMAL', 'SIKAR', 'RANGAREDDI', 'BHIWANI',
          'THE NILGIRIS', 'UNA', 'GANJAM', 'GUNTUR', 'SIWAN', 'VARANASI', 'KARNAL', 'BADGAM', 'FARIDABAD',
          'BANAS KANTHA', 'KANNAUJ', 'KRISHNA', 'DODA', 'HARDOI', 'ANANTANAG', 'JEHANABAD', 'JAGATSINGHAPUR',
          'BAGESHWAR', 'SHIMOGA', 'JHUNJHUNUN', 'CHAMPHAI', 'PITHORAGARH', 'BALESHWAR', 'MORADABAD',
          'WEST GODAVARI', 'BANGALORE RURAL', 'MANDYA', 'CHAMARAJANAGAR', 'AURANGABAD', 'PERAMBALUR', 'PARBHANI',
          'AIZAWL', 'HASSAN', 'CHHATARPUR', 'MAHESANA', 'BANSWARA', 'SHAHJAHANPUR', 'HOSHANGABAD', 'MAU', 'SIRSA',
          'MAHARAJGANJ', 'BIJAPUR', 'SONITPUR', 'SAGAR', 'THOOTHUKKUDI', 'MAHASAMUND', 'BANKURA', 'TAMENGLONG',
          'VISAKHAPATNAM', 'DINDORI', 'DHALAI', 'PRAKASAM', 'MUZAFFARPUR', 'DURG', 'JAJAPUR', 'PONDICHERRY',
          'SAMBALPUR', 'BALAGHAT', 'MALKANGIRI', 'NAWADA', 'EAST NIMAR', 'ETAH', 'SHEIKHPURA', 'HARDA', 'BUXAR',
          'DUNGARPUR', 'KUSHINAGAR', 'LUDHIANA', 'RAICHUR', 'UTTARKASHI', 'NALBARI', 'DHENKANAL', 'JHANSI',
          'UPPER SUBANSIRI', 'SEHORE', 'YAMUNANAGAR', 'AMBEDKAR NAGAR', 'IMPHAL WEST', 'EAST KAMENG', 'TONK',
          'HANUMANGARH', 'JODHPUR', 'JHALAWAR', 'SIROHI', 'MANSA', 'NALGONDA', 'PURBA CHAMPARAN', 'NAWANSHAHR',
          'SUNDARGARH', 'JHABUA', 'BHANDARA', 'BASTI', 'WEST KHASI HILLS', 'KARAIKAL', 'SARAN', 'JHAJJAR', 'CHANDRAPUR',
          'NAMAKKAL', 'SOUTH TRIPURA', 'SAHARANPUR', 'BETUL', 'LALITPUR', 'HAVERI', 'PANNA', 'GANGANAGAR', 'KUPWARA',
          'SOUTH SIKKIM', 'KAPURTHALA', 'CHENNAI', 'GURGAON', 'SHEOPUR', 'EAST SIANG', 'JABALPUR', 'RAJSAMAND',
          'CACHAR', 'KOPPAL', 'TIRAP', 'BOKARO', 'JAISALMER', 'KOLAR', 'SONAPUR', 'KURUKSHETRA', 'AHMADABAD',
          'ALLAHABAD', 'DADRA & NAGAR HAVELI', 'PURNIA', 'NADIA', 'AURAIYA', 'KACHCHH', 'DINDIGUL', 'HAMIRPUR',
          'WEST KAMENG', 'MAINPURI', 'REWA', 'UTTAR DINAJPUR', 'BHIND', 'SITAMARHI', 'ANANTAPUR', 'MANDLA',
          'PASHCHIMI SINGHBHUM', 'PULWAMA', 'WOKHA', 'SIRMAUR', 'SURENDRANAGAR', 'PASHCHIM CHAMPARAN', 'UJJAIN',
          'BASTAR', 'BARABANKI', 'JHARSUGUDA', 'GIRIDIH', 'NARSIMHAPUR', 'KHEDA', 'NAGAON', 'BAUDH', 'ARARIA',
          'BARAN', 'NANDURBAR', 'VILUPPURAM', 'EAST SIKKIM', 'DAMOH', 'KANPUR DEHAT', 'WEST GARO HILLS', 'PATIALA',
          'BATHINDA', 'GONDA', 'JAMMU', 'CHITTAURGARH', 'MOKOKCHUNG', 'BHAGALPUR', 'SANT RAVIDAS NAGAR', 'HATHRAS',
          'HARDWAR', 'MUKTSAR', 'LUNGLEI', 'KATNI', 'ANUGUL', 'KULLU', 'BIDAR', 'MADHEPURA', 'CHANGLANG', 'GUMLA',
          'MUZAFFARNAGAR', 'JIND', 'GADCHIROLI', 'GHAZIPUR', 'LOHIT', 'MAHENDRAGARH', 'DARJILING', 'BALRAMPUR',
          'INDORE', 'MATHURA', 'KENDRAPARA', 'SRIKAKULAM', 'DEBAGARH', 'MORENA', 'RATLAM', 'PANCHKULA', 'KODARMA',
          'SONIPAT', 'NAGPUR', 'PRATAPGARH', 'WEST TRIPURA', 'KENDUJHAR', 'NARMADA', 'GARHWAL', 'BEGUSARAI',
          'SAHIBGANJ', 'MUNGER', 'JANJGIR-CHAMPA', 'CHATRA', 'NALANDA', 'MEDINIPUR', 'SABAR KANTHA', 'GWALIOR',
          'FAIZABAD', 'KOKRAJHAR', 'BALLIA', 'LAKHISARAI', 'THIRUVARUR', 'RAJKOT', 'WEST SIANG', 'PURI', 'CHANDIGARH',
          'COIMBATORE', 'JAIPUR', 'KHAGARIA', 'KISHANGANJ', 'LUCKNOW', 'SIDDHARTHNAGAR']