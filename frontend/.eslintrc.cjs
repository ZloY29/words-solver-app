module.exports = {
  root: true,
  env: {
    browser: true,
    es2022: true,
    node: true,
  },
  extends: ["eslint:recommended", "plugin:vue/vue3-essential"],
  parserOptions: {
    ecmaVersion: "latest",
    sourceType: "module",
  },
  rules: {
    // В маленьких приложениях однословные имена - норм
    "vue/multi-word-component-names": "off",
  },
  ignorePatterns: ["dist/"],
};
