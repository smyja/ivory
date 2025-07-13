declare module 'diff' {
  interface Change {
    count?: number;
    added?: boolean;
    removed?: boolean;
    value: string;
  }

  export function diffLines(oldStr: string, newStr: string, options?: any): Change[];
  export function diffChars(oldStr: string, newStr: string, options?: any): Change[];
  export function diffWords(oldStr: string, newStr: string, options?: any): Change[];
  export function diffWordsWithSpace(oldStr: string, newStr: string, options?: any): Change[];
  export function diffJson(oldObj: any, newObj: any, options?: any): Change[];
  export function diffArrays(oldArr: any[], newArr: any[], options?: any): Change[];
  export function diffTrimmedLines(oldStr: string, newStr: string, options?: any): Change[];
  export function diffSentences(oldStr: string, newStr: string, options?: any): Change[];
  export function diffCss(oldStr: string, newStr: string, options?: any): Change[];
}
