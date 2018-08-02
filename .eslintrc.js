module.exports = {
    "extends": "airbnb-base",
    "rules":{
        "linebreak-style": 0,
        "no-console": 0,
        "no-plusplus": 0,
        "no-loop-func": 0,
        "no-param-reassign": 0,
        "no-await-in-loop": 0,
        "prefer-arrow-callback": 1,
        "array-callback-return": 0,
        "no-unused-vars": 1,
        "no-nested-ternary": 0,
        "prefer-template": 0,
        "object-curly-newline": 0,
        "import/newline-after-import": 1,
        "space-before-function-paren": 0,
        "no-unused-expressions": ["error", {
            "allowTernary": true
        }],    
        "arrow-body-style": ["off", "as-needed"],
        "max-len": ["warn", {
            "code": 120,
            "ignoreComments": true
          }],
        "comma-dangle": ["warn", {
            arrays: 'never',
            objects: 'never',
            imports: 'never',
            exports: 'never',
            functions: 'never'
            }]    
    }, 
};