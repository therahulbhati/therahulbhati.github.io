"use strict";

/** Executable loader for this submission's alphabet + ordered-merge JSON. */
class FaithfulBPETokenizer {
  constructor(alphabet, merges) {
    this.alphabet = alphabet.slice();
    this.merges = merges.map(pair => pair.slice());
    this.byteBase = this.alphabet.length;
    this.vocab = this.alphabet.slice();
    for (let byte = 0; byte < 256; byte++) {
      this.vocab.push(`<0x${byte.toString(16).toUpperCase().padStart(2, "0")}>`);
    }
    for (const [left, right] of this.merges) {
      this.vocab.push(this.vocab[left] + this.vocab[right]);
    }
    this.charToId = new Map(this.alphabet.map((char, id) => [char, id]));
    this.rank = new Map(
      this.merges.map((pair, index) => [`${pair[0]},${pair[1]}`, index])
    );
    this.utf8Encoder = new TextEncoder();
    this.utf8Decoder = new TextDecoder("utf-8", { fatal: false });
    this.wordCache = new Map();
  }

  encodeWord(word) {
    const cached = this.wordCache.get(word);
    if (cached !== undefined) return cached;

    const ids = [];
    for (const char of word) {
      if (this.charToId.has(char)) {
        ids.push(this.charToId.get(char));
      } else {
        for (const byte of this.utf8Encoder.encode(char)) {
          ids.push(this.byteBase + byte);
        }
      }
    }

    while (ids.length >= 2) {
      let bestRank = Infinity;
      let bestPosition = -1;
      for (let position = 0; position < ids.length - 1; position++) {
        const mergeRank = this.rank.get(`${ids[position]},${ids[position + 1]}`);
        if (mergeRank !== undefined && mergeRank < bestRank) {
          bestRank = mergeRank;
          bestPosition = position;
        }
      }
      if (bestPosition < 0) break;
      ids.splice(bestPosition, 2, this.alphabet.length + 256 + bestRank);
    }

    this.wordCache.set(word, ids);
    return ids;
  }

  encode(text) {
    const ids = [];
    for (const word of splitAssignmentWords(text)) {
      ids.push(...this.encodeWord(word));
    }
    return ids;
  }

  decode(ids) {
    const output = [];
    let byteBuffer = [];
    const flushBytes = () => {
      if (byteBuffer.length) {
        output.push(this.utf8Decoder.decode(Uint8Array.from(byteBuffer)));
        byteBuffer = [];
      }
    };

    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= this.vocab.length) {
        throw new RangeError(`Invalid token id: ${id}`);
      }
      if (id >= this.byteBase && id < this.byteBase + 256) {
        byteBuffer.push(id - this.byteBase);
      } else {
        flushBytes();
        output.push(this.vocab[id]);
      }
    }
    flushBytes();
    return output.join("");
  }
}

function characterPolicy(char) {
  const frozen = DATA.character_policy[String(char.codePointAt(0))];
  if (frozen !== undefined) return { wordlike: frozen[0], whitespace: frozen[1] };
  return {
    wordlike: /[\p{L}\p{M}\p{N}]/u.test(char),
    whitespace: /\s/u.test(char),
  };
}

function splitAssignmentWords(text) {
  const words = [];
  let current = "";
  for (const char of text) {
    if (characterPolicy(char).whitespace) {
      if (current) words.push(current);
      current = "";
    } else {
      current += char;
    }
  }
  if (current) words.push(current);
  return words;
}

function visibleText(text) {
  return Array.from(text).filter(char => !characterPolicy(char).whitespace).join("");
}

function faithfulUnits(text) {
  let count = 0;
  let inWordlikeRun = false;
  for (const char of text) {
    const policy = characterPolicy(char);
    if (policy.wordlike) {
      if (!inWordlikeRun) count++;
    } else if (!policy.whitespace) {
      count++;
    }
    inWordlikeRun = policy.wordlike;
  }
  return count;
}

window.FaithfulBPETokenizer = FaithfulBPETokenizer;
window.visibleText = visibleText;
window.faithfulUnits = faithfulUnits;
window.splitAssignmentWords = splitAssignmentWords;
window.tokenizer = new FaithfulBPETokenizer(DATA.alphabet, DATA.merges);
