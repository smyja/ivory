'use client';
import "@mantine/core/styles.css";
import '@mantine/nprogress/styles.css';
import React from "react";
import { MantineProvider, ColorSchemeScript } from "@mantine/core";
import { ModalsProvider } from '@mantine/modals';

import { theme } from "../theme";
import Provider from '../redux/provider';
import localFont from 'next/font/local'
import classes from "./global.module.css"
import { Notifications } from '@mantine/notifications';
import '@mantine/notifications/styles.css';
import '@mantine/dropzone/styles.css';
import { Setup } from '../components/utils';

const satoshi = localFont({
  src: [
    {
      path: '../public/fonts/Satoshi-Light.woff2',
      weight: '300',
    },
    {
      path: '../public/fonts/Satoshi-Regular.woff2',
      weight: '400',
    },
    {
      path: '../public/fonts/Satoshi-Medium.woff2',
      weight: '500',
    },
    {
      path: '../public/fonts/Satoshi-Bold.woff2',
      weight: '700',
    },
  

  ],
});
export default function RootLayout({ children }: { children: any }) {
  return (
    
    <Provider>
    <html lang="en">
      <head>

        <ColorSchemeScript />
        <link rel="shortcut icon" href="/favicon.svg" />
        <meta
          name="viewport"
          content="minimum-scale=1, initial-scale=1, width=device-width, user-scalable=no"
        />
      </head>
      <body className={classes.body}>
        
        <MantineProvider theme={theme}>  
      
        <ModalsProvider>
          <Setup/>
        <Notifications  autoClose={4000} position="bottom-right" zIndex={1000} />
     
          {children}
        </ModalsProvider>
        </MantineProvider>
      </body>
    </html>
    </Provider>
  );
}
