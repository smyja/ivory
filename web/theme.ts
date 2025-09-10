'use client';

import { createTheme } from '@mantine/core';

export const theme = createTheme({
  fontFamily: 'satoshi, sans-serif',
  primaryColor: 'dark',
  components: {
    Button: {
      defaultProps: {
        radius: 0,
      },
      vars: () => ({
        root: {
          '--button-radius': '0px',
        },
      }),
    },
  },
});
