const ObjectsToCsv = require("objects-to-csv");
const fs = require("fs");
const parse = require("csv-parse/lib/sync");
const axios = require("axios");
const fastcsv = require("fast-csv");
const striptags = require("striptags");

const log = console.log;

// this code add descr to challenge

const getDescr = async (id) => {
  const { data } = await axios.get(
    `https://api.topcoder.com/v5/challenges/${id}`,
    {
      headers: {
        "User-Agent": "PostmanRuntime/7.28.4",
        Accept: "*/*",
        "Accept-Encoding": "gzip, deflate, br",
        Connection: "keep-alive",
      },
    }
  );

  return (tagStripData = striptags(
    data["description"]
      .replace(/\n{1,}/g, "")
      .replace(/\s{2,}/g, " ")
      .replace(/ {2,}/g, " ")
      .replace(/\&nbsp;/g, "")
  ));
};

const allMemCsvData = fs.readFileSync(`merged dataset.csv`).toString();

//    calling the npm package and save to records
const memRecords = parse(allMemCsvData, {
  columns: true,
  skip_empty_lines: true,
});

const allDataWithAdd = [];

(async () => {
  let i = 0;
  for (let member of memRecords) {
    try {
      if (i > 20) break;
      const tempMember = { ...member };
      const descr = await getDescr(tempMember["id"]);

      member["challenge description"] = descr;
      log(`${member["challengeId"]}`);
      allDataWithAdd.push(member);
      i++;
    } catch (error) {
      log(`skip due to error`);
    }
  }
  log("saving ....");
  const ws = fs.createWriteStream(
    "scraped_data merge file with chall descr data.csv"
  );
  fastcsv.write(allDataWithAdd, { headers: true }).pipe(ws);
})();

// // this code add descr to members

// const getDescr = async (handle) => {
//   const { data } = await axios.get(
//     `https://api.topcoder.com/v5/members/${handle}`,
//     {
//       headers: {
//         "User-Agent": "PostmanRuntime/7.28.4",
//         Accept: "*/*",
//         "Accept-Encoding": "gzip, deflate, br",
//         Connection: "keep-alive",
//       },
//     }
//   );

//   const { description } = data;

//   return description;
// };

// //   const memDataLocation = `./topcoder/scraped_data member ${item}.csv`;
// const allMemCsvData = fs.readFileSync(`merged dataset.csv`).toString();

// //    calling the npm package and save to records
// const memRecords = parse(allMemCsvData, {
//   columns: true,
//   skip_empty_lines: true,
// });

// const allDataWithAdd = [];

// (async () => {
//   // let i = 0;
//   for (let member of memRecords) {
//     try {
//       // if (i > 20) break;
//       const tempMember = { ...member };
//       const descr = await getDescr(tempMember["handle"]);

//       member["member description"] = descr;
//       log(`${member["challengeId"]} ${member["handle"]}`);
//       allDataWithAdd.push(member);
//       // i++;
//     } catch (error) {
//       log(`skip due to error`);
//     }
//   }
//   log("saving ....");
//   const ws = fs.createWriteStream(
//     "scraped_data merge file with member descr data.csv"
//   );
//   fastcsv.write(allDataWithAdd, { headers: true }).pipe(ws);
// })();

// code to merge member data and challenge data

// const challengeFeatures = "./Dataset/Dataset Task features.csv";
// const memberFeatures = "./DataSet/Dataset developer features.csv";

// const allMemCsvData = fs.readFileSync(memberFeatures).toString();
// const allChallCsvData = fs.readFileSync(challengeFeatures).toString();

// //    calling the npm package and save to records
// const memRecords = parse(allMemCsvData, {
//   columns: true,
//   skip_empty_lines: true,
// });

// const challRecords = parse(allChallCsvData, {
//   columns: true,
//   skip_empty_lines: true,
// });

// // log(challRecords);

// const mergeList = memRecords.map((mem) => {
//   const chall = challRecords.find((item) => item.id === mem["challengeId"]);
//   if (chall) {
//     return { ...mem, ...chall };
//   }
// });

// const ws = fs.createWriteStream("merged dataset ff.csv");
// fastcsv.write(mergeList, { headers: true }).pipe(ws);
