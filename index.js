import fs from 'fs';
import csv from 'csv-parser';

class DecisionTree {
  constructor({ col = -1, value = null, trueBranch, falseBranch, results = null, summary = {} } = {}) {
    this.col = col;
    this.value = value;
    this.trueBranch = trueBranch;
    this.falseBranch = falseBranch;
    this.results = results;
    this.summary = summary;
  }
}

function divideSet(rows, column, value) {
  let splittingFunction = null;
  if (isNumber(value)) {
    splittingFunction = row => row[column] >= value;
  } else {
    splittingFunction = row => row[column] === value;
  }
  const list1 = rows.filter(row => splittingFunction(row));
  const list2 = rows.filter(row => !splittingFunction(row));
  return [list1, list2];
}

function uniqueCounts(rows) {
  const results = {};
  for (const row of rows) {
    const r = row[row.length - 1];
    if (!results.hasOwnProperty(r)) {
      results[r] = 0;
    }
    results[r] += 1;
  }
  return results;
}

function entropy(rows) {
  const results = uniqueCounts(rows);
  let entr = 0.0;
  for (const r in results) {
    const p = results[r] / rows.length;
    entr -= p * Math.log2(p);
  }
  return entr;
}

function gini(rows) {
  const total = rows.length;
  const counts = uniqueCounts(rows);
  let imp = 0.0

  for (const k1 in counts) {
    const p1 = counts[k1] / total;
    for (const k2 in counts) {
      if (k1 === k2) {
        continue;
      }
      const p2 = counts[k2] / total;
      imp += p1 * p2
    }
  }
  return imp;
}

function sum(array) {
  return array.reduce((acc, value) => {
    acc += value;
    return acc;
  }, 0);
}

function isNumber(n) {
  return typeof n === 'number' && !Number.isNaN(n);
}

function variance(rows) {
  if (!rows.length) {
    return 0;
  }

  const data = rows.map(row => row[row.length - 1]);
  if (data.find(p => !isNumber(p))) {
    throw new Error('Cannot use variance function when the target column is a string.');
  }

  const mean = sum(data) / data.length;

  return sum(data.map(d => (d - mean) ** 2)) / data.length;
}

function growDecisionTreeFrom(rows, evaluationFunction = EvaluationFunction.ENTROPY) {
  if (!rows.length) {
    return new DecisionTree;
  }

  const currentScore = evaluationFunction(rows);
  if (Number.isNaN(currentScore)) {
    throw new Error('Something went wrong. Check the evaluation function: ' + evaluationFunction.name);
  }

  let bestGain = 0.0;
  let bestAttribute = null;
  let bestSets = null;

  const columnCount = rows[0].length - 1; // last column is the result/target column

  for (let col = 0; col < columnCount; col++) {
    const columnValues = rows.map(row => row[col]);
    const lsUnique = Array.from(new Set(columnValues)).sort((a, b) => a - b);

    for (const value of lsUnique) {
      const [set1, set2] = divideSet(rows, col, value);

      // Gain -- Entropy or Gini
      const p = set1.length / rows.length;
      const gain = currentScore - (p * evaluationFunction(set1)) - (1 - p) * evaluationFunction(set2);
      if (gain > bestGain && set1.length && set2.length) {
        bestGain = gain;
        bestAttribute = [col, value];
        bestSets = [set1, set2];
      }
    }
  }

  const dcY = {
    impurity: currentScore.toPrecision(3),
    samples: rows.length
  };

  if (bestGain > 0) {
    return new DecisionTree({
      col: bestAttribute[0],
      value: bestAttribute[1],
      trueBranch: growDecisionTreeFrom(bestSets[0], evaluationFunction),
      falseBranch: growDecisionTreeFrom(bestSets[1], evaluationFunction),
      summary: dcY
    });
  } else {
    return new DecisionTree({
      results: uniqueCounts(rows),
      summary: dcY
    })
  }
}

function prune(tree, minGain, evaluationFunction = entropy, notify = false) {
  if (!tree.trueBranch.results) {
    prune(tree.trueBranch, minGain, evaluationFunction, notify);
  }
  if (!tree.falseBranch.results) {
    prune(tree.falseBranch, minGain, evaluationFunction, notify);
  }

  if (tree.trueBranch.results && tree.falseBranch.results) {
    let tb = [];
    let fb = [];

    for (const [v, c] of Object.entries(tree.trueBranch.results)) {
      tb = tb.concat(Array.from({ length: c }, _ => [v]))
    }

    for (const [v, c] of Object.entries(tree.falseBranch.results)) {
      fb = fb.concat(Array.from({ length: c }, _ => [v]))
    }

    const p = tb.length / (tb.length + fb.length);
    const delta = evaluationFunction(tb.concat(fb)) - p * evaluationFunction(tb) - (1 - p) * evaluationFunction(fb);

    if (delta < minGain) {
      if (notify) {
        console.log(`A branch was pruned: gain = ${delta}`);
      }
      tree.trueBranch = null;
      tree.falseBranch = null;
      tree.results = uniqueCounts(tb.concat(fb));
    }
  }
}

function classify(observations, tree, dataMissing = false) {
  function classifyWithoutMissingData(observations, tree) {
    if (tree.results) {
      return tree.results;
    }
    const v = observations[tree.col];
    let branch = null;
    if (isNumber(v)) {
      if (v >= tree.value) {
        branch = tree.trueBranch;
      } else {
        branch = tree.falseBranch;
      }
    } else {
      if (v === tree.value) {
        branch = tree.trueBranch;
      } else {
        branch = tree.falseBranch;
      }
    }
    return classifyWithoutMissingData(observations, branch);
  }

  function classifyWithMissingData(observations, tree) {
    if (tree.results) {
      return tree.results;
    }
    const v = observations[tree.col]
    if (v === null) {
      const tr = classifyWithMissingData(observations, tree.trueBranch);
      const fr = classifyWithMissingData(observations, tree.falseBranch);

      const tCount = sum(Object.values(tr));
      const fCount = sum(Object.values(fr));

      const tw = tCount / (tCount + fCount);
      const fw = fCount / (tCount + fCount);

      const result = {};
      for (const [k, v] of Object.entries(tr)) {
        if (!result.hasOwnProperty(k)) {
          result[k] = 0;
        }
        result[k] += v * tw;
      }
      for (const [k, v] of Object.entries(fr)) {
        if (!result.hasOwnProperty(k)) {
          result[k] = 0;
        }
        result[k] += v * fw;
      }

      return result;
    } else {
      let branch = null;
      if (isNumber(v)) {
        if (v >= tree.value) {
          branch = tree.trueBranch;
        } else {
          branch = tree.falseBranch;
        }
      } else {
        if (v === tree.value) {
          branch = tree.trueBranch;
        } else {
          branch = tree.falseBranch;
        }
      }
      return classifyWithMissingData(observations, branch);
    }
  }

  if (dataMissing) {
    return classifyWithMissingData(observations, tree);
  } else {
    return classifyWithoutMissingData(observations, tree);
  }
}

function plot(decisionTree, dcHeadings = {}, indent = '') {
  if (decisionTree.results) {
    const lsX = Object.entries(decisionTree.results);
    lsX.sort((a, b) => a[0] - b[0]);
    const szY = lsX.map(([x, y]) => `(${x}: ${y})`).join(', ');
    return szY;
  } else {
    const szCol = decisionTree.col in dcHeadings ? dcHeadings[decisionTree.col] : `Column ${decisionTree.col}`;
    let decision;
    if (isNumber(decisionTree.value)) {
      decision = `${szCol} >= ${decisionTree.value}`;
    } else {
      decision = `${szCol} == ${decisionTree.value}?`;
    }
    const trueBranch = indent + 'yes -> ' + plot(decisionTree.trueBranch, dcHeadings, indent + '\t\t');
    const falseBranch = indent + 'no -> ' + plot(decisionTree.falseBranch, dcHeadings, indent + '\t\t');
    return decision + '\n' + trueBranch + '\n' + falseBranch;
  }
}

function formatValue(v) {
  const number = Number(v);
  if (!Number.isNaN(number)) {
    return number;
  }
  return v.trim();
}

async function readCSV(filepath, { hasHeaders = true } = {}) {
  return new Promise((resolve, reject) => {
    const result = { headers: [], rows: [] };
    let rowLength = null;
    fs.createReadStream(filepath)
      .pipe(csv({
        headers: !hasHeaders ? false: undefined,
        mapHeaders: ({ header, index }) => {
          result.headers[index] = header;
          return index;
        },
      }))
      .on('data', (data) => {
        if (rowLength === null) {
          rowLength = Object.keys(data).length;
        }
        result.rows.push(Array.from(Object.assign(data, { length: rowLength }), formatValue));
      })
      .on('end', () => resolve(result))
      .on('error', reject);
  })
}

const EvaluationFunction = {
  ENTROPY: entropy,
  GINI: gini,
  VARIANCE: variance,
}

async function main() {
  const example = 2;

  if (example === 1) {
    const { headers, rows } = await readCSV('./tbc.csv', { hasHeaders: false });
    const decisionTree = growDecisionTreeFrom(rows);
    // const decisionTree = growDecisionTreeFrom(rows, EvaluationFunction.GINI); // with Gini
    const result = plot(decisionTree, headers);
    console.log(result);

    console.log(classify(['ohne', 'leicht', 'Streifen', 'normal', 'normal'], decisionTree, false))
    console.log(classify([null, 'leicht', null, 'Flocken', 'fiepend'], decisionTree, true)) // no longer unique

    // Don't forget if you compare the resulting tree with the tree in my presentation: here it is a binary tree!
  } else if (example === 2) {
    const { headers, rows } = await readCSV('./fishiris.csv', { hasHeaders: true });
    const decisionTree = growDecisionTreeFrom(rows, EvaluationFunction.GINI)
    prune(decisionTree, 0.8, undefined, true) // notify, when a branch is pruned (one time in this
    // example)
    const result = plot(decisionTree, headers);
    console.log(result);

    console.log(classify([6.0, 2.2, 5.0, 1.5], decisionTree)) // dataMissing=false is the default setting
    console.log(classify([null, null, null, 1.5], decisionTree, true)) // no longer unique
  }
}

main();
