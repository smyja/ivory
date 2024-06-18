import { useState } from "react";
import { ChangeEvent, FormEvent } from 'react';
import { useRouter } from 'next/navigation';
import { useAppDispatch } from '../redux/hooks';
import { useLoginMutation } from '../redux/features/authApiSlice';
import { setAuth } from '../redux/features/authSlice';
import { notifications } from '@mantine/notifications';
import classes from "./Error.module.css"
interface FormValues {
  email: string;
  password: string;
}

export default function useLogin(form: { setValue: Function, values: any }) {
  const router = useRouter();
  const dispatch = useAppDispatch();
  const [login, { isLoading }] = useLoginMutation();
  const [loginError, setLoginError] = useState(false);
  const onChange = (event: ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    form.setValue(name, value);
  };

  const onSubmit = (values: FormValues, event?: React.FormEvent<HTMLFormElement>) => {
    event?.preventDefault();

    const { email, password } = form.values;

    login({ email, password })
      .unwrap()
      .then(() => {
        dispatch(setAuth());
        notifications.show({
          title: "Successful",
          message: 'Logging in to your account',
          color: 'teal',
          radius:"md",
          loading: true,
        });
        router.push('/dashboard');
      })
      .catch(() => {
        setLoginError(true);
        notifications.show({
          title: "Error",
          message: 'Something went wrong',
          color: 'red',
          radius:"md",
          classNames:classes
     
        });
      });
  }

  return {
    isLoading,
    onChange,
    onSubmit,
    setValue: form.setValue,
    loginError // Add the setValue function to the return value
    
  };
}